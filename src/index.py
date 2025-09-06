import chromadb
import yaml
from typing import List
import numpy as np

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def index_documents(chunks: List[dict], embeddings: np.ndarray):
    """Индексирует чанки в Chroma."""
    config = load_config()
    persist_dir = config['chroma']['persist_directory']
    collection_name = config['chroma']['collection_name']
    
    client = chromadb.PersistentClient(path=persist_dir)
    
    # Удаляем старую коллекцию, если она существует
    try:
        client.delete_collection(name=collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except ValueError:
        print(f"Collection {collection_name} does not exist, creating a new one.")
    
    # Создаем новую коллекцию
    collection = client.create_collection(name=collection_name)
    
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk['content']],
            metadatas=[{"file_name": chunk['file_name'], "chunk_id": chunk['chunk_id']}],
            ids=[f"{chunk['file_name']}_{chunk['chunk_id']}"],
            embeddings=[embeddings[i].tolist()]
        )
    return collection
