"""Application configuration via pydantic-settings (reads from .env)."""
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    hf_token: str = ""
    llama_model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
    mistral_model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"
    embedding_model_id: str = "intfloat/multilingual-e5-large"
    chroma_persist_dir: str = "./data/vectorstore"
    backend_url: str = "http://localhost:8000"
    max_tokens_default: int = 1024
    rag_top_k_default: int = 4
    log_level: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()
