import os
from PyPDF2 import PdfReader
from docx import Document
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_document(file_path):
    """Загружает документ в зависимости от его формата."""
    config = load_config()
    supported_formats = config['data']['supported_formats']
    
    ext = os.path.splitext(file_path)[1].lower().lstrip('.')
    if ext not in supported_formats:
        raise ValueError(f"Формат {ext} не поддерживается. Поддерживаемые форматы: {supported_formats}")

    if ext == "pdf":
        return load_pdf(file_path)
    elif ext == "txt":
        return load_txt(file_path)
    elif ext == "docx":
        return load_docx(file_path)

def load_pdf(file_path):
    """Извлечение текста из PDF."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def load_txt(file_path):
    """Чтение текста из TXT."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def load_docx(file_path):
    """Извлечение текста из DOCX."""
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text])

def ingest_documents(data_dir):
    """Загружает все документы из указанной директории."""
    config = load_config()
    data_dir = config['data']['input_dir']
    documents = []
    for file_name in os.listdir(data_dir):
        if file_name.startswith('.'):  # Игнорируем скрытые файлы, например .DS_Store
            continue
        file_path = os.path.join(data_dir, file_name)
        try:
            text = load_document(file_path)
            documents.append({"file_name": file_name, "content": text})
        except Exception as e:
            print(f"Ошибка при загрузке {file_name}: {e}")
    return documents
