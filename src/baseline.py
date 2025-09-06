from transformers import pipeline
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def baseline_answer(query: str):
    """Генерирует ответ без retrieval."""
    config = load_config()
    model_name = config['generation']['model']
    max_length = config['generation']['max_length']
    temperature = config['generation']['temperature']
    
    generator = pipeline("text-generation", model=model_name)
    prompt = f"Question: {query}\nAnswer:"
    answer = generator(prompt, max_length=max_length, temperature=temperature, num_return_sequences=1)
    return answer[0]['generated_text'].split("Answer:")[-1].strip()
