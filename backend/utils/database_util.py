from database import SessionLocal
from models import Cell, Document, Chunk
from typing import List
import os
from uuid import UUID

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def save_document_chunks(file_ref: str, chunks: List[str]):
    """
    Create a new document entry and save its chunks in the database.
    """
    session = SessionLocal()
    try:
        # get filename from file_ref
        filename = os.path.basename(file_ref)
        # create document entry
        document = Document(file_ref=file_ref, filename=filename)
        session.add(document)
        session.commit()
        session.refresh(document)
        # create chunks
        for index, chunk_text in enumerate(chunks):
            chunk = Chunk(
                document_id=document.id,
                chunk_index=index,
                text=chunk_text
            )
            session.add(chunk)
        session.commit()
        return document.id  # Return the document ID
    except:
        session.rollback()
        raise
    finally:
        session.close()

def get_chunks_by_ids(db, chunk_ids):
    """
    Fetch chunk texts from the database given a list of chunk_ids of format '<document_id>-chunk-<index>'.
    Returns a mapping from chunk_id to chunk text.
    """
    texts = {}
    for cid in chunk_ids:
        try:
            doc_id_str, idx_str = cid.rsplit('-chunk-', 1)
            doc_uuid = UUID(doc_id_str)
            idx = int(idx_str)
        except ValueError:
            continue
        chunk = db.query(Chunk).filter(
            Chunk.document_id == doc_uuid,
            Chunk.chunk_index == idx
        ).first()
        if chunk:
            texts[cid] = chunk.text
    return texts


def save_cell(db, row_id: UUID, column_id: UUID, answer: str):
    """
    Save a cell to the database.
    """
    cell = Cell(
        row_id=row_id,
        column_id=column_id,
        answer=answer
    )
    db.add(cell)
    db.commit()
    db.refresh(cell)
    return cell.id