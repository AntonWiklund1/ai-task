from typing import List
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from pydantic import BaseModel
import logging

load_dotenv()

logger = logging.getLogger(__name__)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

class PineconeMetadata(BaseModel):
    document_id: str
    chunk_id: str

class PineconeDocument(BaseModel):
    id: str
    embedding: List[float]
    metadata: PineconeMetadata

def create_pinecone_documents(chunks: List[str], embeddings: List[List[float]], file_ref: str) -> List[PineconeDocument]:
    """
    Creates a list of PineconeDocument objects from chunks, embeddings, and a file reference.
    """
    documents = []
    for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
        document_id = f"{file_ref}-chunk-{i}"
        metadata = PineconeMetadata(
            document_id=file_ref,
            chunk_id=document_id
        )
        documents.append(PineconeDocument(id=document_id, embedding=embedding, metadata=metadata))
    logger.info(f"Created {len(documents)} Pinecone documents for file_ref: {file_ref}")
    return documents

def upsert_documents(
    documents: List[PineconeDocument],
    batch_size: int = 100
) -> None:
    """
    Upsert a list of PineconeDocument into the Pinecone index
    in batches of `batch_size`.
    """
    # prepare the raw list of dicts
    to_upsert = [
        {
            "id": doc.id,
            "values": doc.embedding,
            "metadata": doc.metadata.model_dump()
        }
        for doc in documents
    ]

    # slice into batches and send
    for i in range(0, len(to_upsert), batch_size):
        batch = to_upsert[i : i + batch_size]
        index.upsert(vectors=batch)

    logger.info(f"✅ Upserted {len(documents)} vectors in {((len(documents)-1)//batch_size)+1} batches")

def query_pinecone(query_embedding: List[float], allowed_docs: List[str]):
    response = index.query(
        vector=query_embedding,
        top_k=10,
        include_metadata=True,
        filter={
            "document_id": {
                "$in": allowed_docs
            }
        }
    )
    return response
