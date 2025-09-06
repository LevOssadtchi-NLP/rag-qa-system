import pytest
import os
from src.ingest import load_document

def test_load_txt():
    with open("test.txt", "w", encoding="utf-8") as f:
        f.write("Test content")
    text = load_document("test.txt")
    assert text == "Test content"
    os.remove("test.txt")

def test_unsupported_format():
    with pytest.raises(ValueError):
        load_document("test.jpg")
