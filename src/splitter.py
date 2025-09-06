from typing import List

def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Разбивает текст на чанки с учетом перекрытия."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    
    return chunks

def split_documents(documents: List[dict], chunk_size: int = 500, overlap: int = 50) -> List[dict]:
    """Разбивает все документы на чанки."""
    chunked_docs = []
    for doc in documents:
        chunks = split_text(doc['content'], chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            chunked_docs.append({
                "file_name": doc['file_name'],
                "chunk_id": i,
                "content": chunk
            })
    return chunked_docs
