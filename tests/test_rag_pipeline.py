import pytest
from src.rag_pipeline import rag_pipeline
from src.utils import load_config

def test_rag_pipeline():
    config = load_config()
    query = "What is the capital of France?"
    answer, metadatas = rag_pipeline(query)
    assert isinstance(answer, str)
    assert isinstance(metadatas, list)
