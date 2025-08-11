from pathlib import Path

BASE_DIR = Path("D:/VisualSC/MyChatbotRAG")

DATA_SOURCE_DIR = BASE_DIR / "data_source"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
CHAT_HISTORY_FILE = BASE_DIR / "chat_history" / "chat_log.json"
OLLAMA_API_URL = "http://localhost:11434"
