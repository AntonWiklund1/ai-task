from typing import List
from pydantic import BaseModel, Field
from uuid import UUID
import uvicorn
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import logging
from sqlalchemy.orm import Session
from fastapi import Depends

# SQLAlchemy setup
from utils.rag_pipeline import rag_pipeline
from utils.pinecone_util import upsert_documents, create_pinecone_documents
from database import engine, Base
from utils.database_util import save_document_chunks, get_db, save_cell
from models import Column as ColumnModel, Row as RowModel, Document as DocumentModel, row_documents
from utils.text import get_text_from_file, RecursiveTokenChunker
from utils.embedding import VoyageEmbeddings
from utils.llm import get_answer

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

class UploadDocumentResponse(BaseModel):
    message: str
    document_id: UUID

@app.post("/upload-document", response_model=UploadDocumentResponse)
async def upload_document(request: UploadDocumentRequest, db: Session = Depends(get_db)):
    try:
        # Check if document with this file_ref already exists
        existing_document = db.query(DocumentModel).filter(DocumentModel.file_ref == request.file_ref).first()
        if existing_document:
            return {"message": "Document already uploaded", "document_id": existing_document.id}

        text = get_text_from_file(request.file_ref)
        logger.info(f"got text {text[:100]}")
        chunks = chunker.split_text(text)
        logger.info(f"got chunks {len(chunks)}")
        embeddings = embedding_model.get_embeddings(chunks, input_type="document")
        logger.info(f"got embeddings {len(embeddings)}")   

        # save the chunks to the database
        document_id = save_document_chunks(request.file_ref, chunks)

        # create pinecone documents
        pinecone_documents = create_pinecone_documents(chunks, embeddings, str(document_id))

        # upsert the embedding to pinecone
        upsert_documents(pinecone_documents)

        return {"message": "Document uploaded successfully", "document_id": document_id}
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class ColumnCreateRequest(BaseModel):
    label: str = Field(..., example="Agreement date")
    prompt: str = Field(..., example="What is the date when the agreement went into force?")
    format: str = Field(..., example="date")  # must be one of text|date|boolean|currency

class ColumnCreateResponse(BaseModel):
    column_id: UUID

@app.post("/columns", response_model=ColumnCreateResponse)
def create_column(req: ColumnCreateRequest, db: Session = Depends(get_db)):
    try:
        # Validate format
        if req.format not in {f.value for f in ColumnModel.format.type.enum_class}:  
            raise HTTPException(400, "Invalid format")

        col = ColumnModel(
            label=req.label,
            prompt=req.prompt,
            format=req.format
        )
        db.add(col)
        db.commit()
        db.refresh(col)

        return ColumnCreateResponse(column_id=col.id)
    except Exception as e:
        logger.error(f"Error creating column: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class RowCreateRequest(BaseModel):
    document_ids: List[UUID]

class RowCreateResponse(BaseModel):
    row_id: UUID

@app.post("/rows", response_model=RowCreateResponse)
def create_row(req: RowCreateRequest, db: Session = Depends(get_db)):
    # Check for duplicates in the input document_ids list
    if len(req.document_ids) != len(set(req.document_ids)):
        seen = set()
        duplicates = {str(doc_id) for doc_id in req.document_ids if doc_id in seen or seen.add(doc_id)}
        raise HTTPException(
            status_code=400,
            detail=f"Duplicate document IDs provided: {', '.join(duplicates)}"
        )

    # Validate that all document_ids exist
    requested_ids_set = set(req.document_ids)  # Use a set of the requested IDs
    db_result = db.query(DocumentModel.id).filter(DocumentModel.id.in_(requested_ids_set)).all()
    existing_db_ids = {row.id for row in db_result}

    missing_ids = requested_ids_set - existing_db_ids
    if missing_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Documents not found: {', '.join(str(m) for m in missing_ids)}"
        )

    # Create the new Row
    row = RowModel()
    db.add(row)
    db.flush()   # populates row.id without committing

    # Link documents via the association table
    link_values = [
        {"row_id": row.id, "document_id": doc_id}
        for doc_id in req.document_ids
    ]
    db.execute(row_documents.insert().values(link_values))

    # Commit and return
    db.commit()
    return RowCreateResponse(row_id=row.id)

class AnswerItem(BaseModel):
    row_id: str
    column_id: str

class AnswerRequest(BaseModel):
    items: List[AnswerItem]

class AnswerResponseItem(BaseModel):
    row_id: str
    column_id: str
    answer: str
    cell_id: UUID

class AnswerResponse(BaseModel):
    results: List[AnswerResponseItem]

@app.post("/answer", response_model=AnswerResponse)
async def answer(request: AnswerRequest, db: Session = Depends(get_db)):
    try:
        results: List[AnswerResponseItem] = []
        for item in request.items:
            column = db.query(ColumnModel).get(item.column_id)
            if not column:
                raise HTTPException(404, f"Column not found: {item.column_id}")

            doc_ids_uuids = [
                rd.document_id
                for rd in db.query(row_documents)
                         .filter_by(row_id=item.row_id)
                         .all()
            ]
            doc_ids = [str(uuid_obj) for uuid_obj in doc_ids_uuids] # used for restricitng what to retrieve from pinecone

            n_documents = len(doc_ids)
            logger.info(f"doc_ids for row {item.row_id}: {doc_ids}")

            if not doc_ids:
                raise HTTPException(404, f"Row {item.row_id} has no documents")

            rag_answer = await rag_pipeline(column.prompt, doc_ids, db, embedding_model)
            answer_text = get_answer(column.prompt, rag_answer, column.format, n_documents)
            cell_id = save_cell(db, item.row_id, item.column_id, answer_text)
            results.append(AnswerResponseItem(
                row_id=item.row_id,
                column_id=item.column_id,
                answer=answer_text,
                cell_id=cell_id
            ))

        logger.info(f"answers: {results}")
        return {"results": results}
    except Exception as e:
        logger.error(f"Error processing batch answers: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")