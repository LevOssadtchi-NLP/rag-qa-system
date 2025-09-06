import streamlit as st
import os
from src.ingest import ingest_documents, load_document
from src.splitter import split_documents
from src.embed import create_embeddings
from src.index import index_documents
from src.rag_pipeline import rag_pipeline
from src.baseline import baseline_answer
from src.utils import load_config

st.title("RAG QA System")

# Загрузка конфигурации
config = load_config()

# Загрузка документов
st.header("Upload Documents")
uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=config['data']['supported_formats'])

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(config['data']['input_dir'], uploaded_file.name)
        os.makedirs(config['data']['input_dir'], exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    # Индексация
    if st.button("Index Documents"):
        documents = ingest_documents(config['data']['input_dir'])
        chunks = split_documents(documents)
        embeddings = create_embeddings(chunks)
        collection = index_documents(chunks, embeddings)
        st.success("Documents indexed successfully!")

# Запрос и ответ
st.header("Ask a Question")
query = st.text_input("Enter your question:")
if query:
    st.subheader("RAG Answer")
    answer, metadatas = rag_pipeline(query)
    st.write(answer)
    st.write("Sources:")
    for meta in metadatas:
        st.write(f"- {meta['file_name']} (chunk {meta['chunk_id']})")
    
    st.subheader("Baseline Answer (No Retrieval)")
    baseline = baseline_answer(query)
    st.write(baseline)
