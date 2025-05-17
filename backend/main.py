from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import logging

# SQLAlchemy setup
from utils.pinecone_util import upsert_documents, create_pinecone_documents
from database import engine, Base
import models
from utils.text import get_text_from_file, RecursiveTokenChunker
from utils.embedding import VoyageEmbeddings

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI()
embedding_model = VoyageEmbeddings()
chunker = RecursiveTokenChunker()

@app.get("/")
def read_root():
    logger.info("Hello, World!")
    return {"message": "Hello, World!"}

class UploadDocumentRequest(BaseModel):
    file_ref: str

@app.post("/upload-document")
async def upload_document(request: UploadDocumentRequest):
    try:
        text = get_text_from_file(request.file_ref)
        logger.info(f"got text {text[:100]}")
        chunks = chunker.split_text(text)
        logger.info(f"got chunks {len(chunks)}")
        embeddings = await embedding_model.get_embeddings(chunks, input_type="document")
        logger.info(f"got embeddings {len(embeddings)}")   

        # create pinecone documents
        pinecone_documents = create_pinecone_documents(chunks, embeddings, request.file_ref)

        # upsert the embedding to pinecone
        upsert_documents(pinecone_documents)
        return {"message": "Document uploaded successfully"}
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")