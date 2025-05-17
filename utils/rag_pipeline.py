from typing import List
from uuid import UUID
from utils.pinecone_util import query_pinecone, rerank_pinecone_results
from utils.database_util import get_chunks_by_ids
from sqlalchemy.orm import Session

async def rag_pipeline(query: str, doc_ids: List[UUID], db: Session , embedding_model) -> str:

    query_embedding = embedding_model.get_embeddings([query], input_type="query")

    pinecone_results = query_pinecone(query_embedding, doc_ids)

    # fetch chunk texts from database based on pinecone matches
    matches = pinecone_results.get("matches", [])
    # sort matches by score descending
    matches = sorted(matches, key=lambda m: m["score"], reverse=True)
    chunk_ids = [m["metadata"]["chunk_id"] for m in matches]
    chunk_texts = get_chunks_by_ids(db, chunk_ids)
    response_chunks = [
        {"chunk_id": cid, "score": m["score"], "text": chunk_texts.get(cid, "")}  
        for m, cid in zip(matches, chunk_ids)
    ]
    # sort response_chunks by score descending
    response_chunks = sorted(response_chunks, key=lambda c: c["score"], reverse=True)

    rerank_result = rerank_pinecone_results(query, [c["text"] for c in response_chunks])
    
    if rerank_result and isinstance(rerank_result, list):
        all_texts = [doc.get("text", "") for doc in rerank_result if isinstance(doc, dict)]
        return "\n".join(all_texts)
    return ""
