import warnings
import logging
warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)

from src.ingest import ingest_documents
from src.splitter import split_documents
from src.embed import create_embeddings
from src.index import index_documents
from src.rag_pipeline import rag_pipeline
from src.utils import setup_logging

def main():
    # Настройка логирования
    logger = setup_logging()
    logger.info("Starting RAG pipeline...")

    # Загрузка документов
    logger.info("Ingesting documents...")
    documents = ingest_documents("data/")
    
    # Разбиение на чанки
    logger.info("Splitting documents...")
    chunks = split_documents(documents)
    
    # Создание эмбеддингов
    logger.info("Creating embeddings...")
    embeddings = create_embeddings(chunks)
    
    # Индексация
    logger.info("Indexing documents...")
    collection = index_documents(chunks, embeddings)
    
    # Пример запроса
    query = "Радиус Шварцшильда - это"
    logger.info(f"Query: {query}")
    answer, metadatas = rag_pipeline(query)
    
    # Вывод результата
    print(f"Query: {query}")
    print(f"Answer: {answer}")
    print("Sources:")
    for meta in metadatas:
        print(f"- {meta['file_name']} (chunk {meta['chunk_id']})")

if __name__ == "__main__":
    main()
