import logging
import yaml

def setup_logging():
    """Настраивает логирование."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("rag_system.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def load_config(config_path="config.yaml"):
    """Загружает конфигурацию."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
