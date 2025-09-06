from sentence_transformers import SentenceTransformer
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def create_embeddings(chunks: list, model_name: str = None, device: str = None):
    """Создает векторные представления для чанков."""
    config = load_config()
    model_name = model_name or config['embeddings']['model']
    device = device or config['embeddings']['device']
    
    model = SentenceTransformer(model_name, device=device)
    texts = [chunk['content'] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    return embeddings
