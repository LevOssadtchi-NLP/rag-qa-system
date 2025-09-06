from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer
import chromadb  # Добавляем импорт chromadb
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def retrieve_relevant_chunks(query: str, collection, model_name: str, device: str, top_k: int = 2):
    """Извлекает релевантные чанки из Chroma."""
    model = SentenceTransformer(model_name, device=device)
    query_embedding = model.encode([query])[0]
    
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    
    return results['documents'][0], results['metadatas'][0]

def generate_answer(query: str, context: str, model_name: str, max_new_tokens: int, temperature: float):
    """Генерирует ответ на основе контекста."""
    # Инициализация токенизатора
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Формирование промпта
    prompt = f"Question: {query}\nContext: {context}\nAnswer:"
    
    # Подсчет токенов во входном тексте
    input_tokens = tokenizer(prompt, return_tensors="pt", truncation=False)
    input_length = input_tokens['input_ids'].shape[1]
    
    # Ограничение длины контекста, чтобы оставить место для ответа
    max_input_tokens = 400  # Оставляем место для max_new_tokens
    if input_length > max_input_tokens:
        # Обрезаем контекст
        context_tokens = tokenizer(context, return_tensors="pt", truncation=False)['input_ids']
        context_length = context_tokens.shape[1]
        question_tokens = tokenizer(f"Question: {query}\nAnswer:", return_tensors="pt", truncation=False)['input_ids'].shape[1]
        available_context_tokens = max_input_tokens - question_tokens
        if available_context_tokens < context_length:
            truncated_context = tokenizer.decode(context_tokens[0, :available_context_tokens], skip_special_tokens=True)
            prompt = f"Question: {query}\nContext: {truncated_context}\nAnswer:"
    
    # Генерация ответа
    generator = pipeline("text-generation", model=model_name)
    answer = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        num_return_sequences=1,
        truncation=True
    )
    return answer[0]['generated_text'].split("Answer:")[-1].strip()

def rag_pipeline(query: str):
    """Основной RAG пайплайн."""
    config = load_config()
    model_name = config['embeddings']['model']
    device = config['embeddings']['device']
    gen_model = config['generation']['model']
    max_new_tokens = config['generation']['max_new_tokens']
    temperature = config['generation']['temperature']
    persist_dir = config['chroma']['persist_directory']
    collection_name = config['chroma']['collection_name']
    
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(name=collection_name)
    
    # Извлечение релевантных чанков
    documents, metadatas = retrieve_relevant_chunks(query, collection, model_name, device)
    context = "\n".join(documents)
    
    # Отладочный вывод контекста
    print(f"Retrieved context: {context}")
    
    # Генерация ответа
    answer = generate_answer(query, context, gen_model, max_new_tokens, temperature)
    return answer, metadatas
