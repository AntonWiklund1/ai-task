from typing import List
from uuid import UUID
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

def create_pinecone_documents(chunks: List[str], embeddings: List[List[float]], document_id: UUID) -> List[PineconeDocument]:
    """
    Creates a list of PineconeDocument objects from chunks, embeddings, and a file reference.
    """
    documents = []
    for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"{document_id}-chunk-{i}"
        metadata = PineconeMetadata(
            document_id=document_id,
            chunk_id=chunk_id
        )
        documents.append(PineconeDocument(id=chunk_id, embedding=embedding, metadata=metadata))
    logger.info(f"Created {len(documents)} Pinecone documents for document_id: {document_id}")
    return documents

def upsert_documents(
    documents: List[PineconeDocument],
    batch_size: int = 100
) -> None:
    """
    Upsert a list of PineconeDocument into the Pinecone index
    in batches of `batch_size`.
    """
    try:
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
    except Exception as e:
        logger.error(f"Error upserting documents: {e}")
        raise e

def query_pinecone(query_embedding: List[float], allowed_docs: List[str], top_k: int = 50):

    logger.info(f"Querying Pinecone with allowed_docs: {allowed_docs}, top_k: {top_k}")
    
    # Ensure all allowed_docs are strings
    processed_allowed_docs = [str(doc_id) for doc_id in allowed_docs]

    response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter={
            "document_id": {
                "$in": processed_allowed_docs
            }
        }
    )
    return response

def rerank_pinecone_results(prompt: str, documents: List[str]):
    logger.info(f"Reranking Pinecone results")
    rerank_result = pc.inference.rerank(
                    model="pinecone-rerank-v0", 
                    query=prompt,
                    documents=documents,
                    top_n=min(len(documents), 15),
                    return_documents=True,
                    parameters={
                        "truncate": "END"  # Truncate at token limit if needed
                    }
    )
    # Extract the serializable data
    if rerank_result and hasattr(rerank_result, 'data'):
        return [{"score": item.score, "text": item.document.text if hasattr(item.document, 'text') else item.document} for item in rerank_result.data]
    return []