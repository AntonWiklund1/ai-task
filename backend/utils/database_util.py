from database import SessionLocal
from models import Document, Row, Chunk, row_documents
from typing import List
import os

def save_document_chunks(file_ref: str, chunks: List[str]):
    """
    Create a new row and document entries, associate them, and save chunks in the database.
    """
    session = SessionLocal()
    try:
        # derive filename from file_ref
        filename = os.path.basename(file_ref)
        # create document entry
        document = Document(file_ref=file_ref, filename=filename)
        session.add(document)
        session.commit()
        session.refresh(document)
        # create a row entry
        row = Row(name=filename)
        session.add(row)
        session.commit()
        session.refresh(row)
        # associate the row with the document
        session.execute(row_documents.insert().values(row_id=row.id, document_id=document.id))
        session.commit()
        # create chunks
        for index, chunk_text in enumerate(chunks):
            chunk = Chunk(
                row_id=row.id,
                document_id=document.id,
                chunk_index=index,
                text=chunk_text
            )
            session.add(chunk)
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

