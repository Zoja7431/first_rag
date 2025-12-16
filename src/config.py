"""
config.py — загрузка конфигурации из config.yaml без дублирования значений.

Все конкретные значения (пути, model_name, url, api_key и т.д.)
берутся ТОЛЬКО из config/config.yaml.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


# ---------- Pydantic-модели под структуру config.yaml ----------

class DataConfig(BaseModel):
    pdf_path: str
    books: List[str]


class TextProcessingConfig(BaseModel):
    # text_processing:
    #   chunk_size: 512
    #   chunk_overlap: 50
    #   spacy_model: "ru_core_news_sm"
    chunk_size: int
    overlap_size: int
    spacy_model: str


class EmbeddingsConfig(BaseModel):
    # embeddings:
    #   model_name: "sentence-transformers/all-MiniLM-L6-v2"
    #   embedding_dim: 384
    #   device: "cpu"
    #   model_kwargs:
    #     trust_remote_code: true
    model_name: str
    embedding_dim: int
    device: str
    model_kwargs: Dict[str, Any] = {}  # можно оставить пустой dict, если в YAML нет


class QdrantConfig(BaseModel):
    url: str
    api_key: str = Field(..., description="Cloud API key")  # ← Обязательный для cloud
    collection_name: str
    vector_size: int
    distance: str
    search_limit: int
    port: Optional[int] = None  # ← Для local, опционально

    @property
    def full_url(self) -> str:
        """Cloud: полный url, local: url:port"""
        if self.api_key and ':' in self.url:  # Cloud full URL
            return self.url
        # Добавь Optional[int] = None в класс QdrantConfig
        port = self.port if hasattr(self, 'port') and self.port else 6333
        return f"{self.url}:{port}"

class PromptConfig(BaseModel):
    # prompt:
    #   system_role: "..."
    #   retrieval_template: "..."
    system_role: str
    retrieval_template: str


# Если позже раскомментируешь llm / graph / logging / docker в YAML,
# можно будет добавить сюда опциональные модели:
#
# class LLMConfig(BaseModel):
#     type: str
#     ollama: Optional[OllamaConfig] = None
#     huggingface: Optional[HFConfig] = None
#
# и т.д.


class Settings(BaseModel):
    data: DataConfig
    text_processing: TextProcessingConfig
    embeddings: EmbeddingsConfig
    qdrant: QdrantConfig
    prompt: PromptConfig

    # На будущее (пока в YAML закомментировано):
    # llm: Optional[LLMConfig] = None
    # graph: Optional[GraphConfig] = None
    # logging: Optional[LoggingConfig] = None
    # docker: Optional[DockerConfig] = None


# ---------- Функции загрузки / доступа к конфигу ----------

_config: Optional[Settings] = None


def load_config(path: str = "config/config.yaml") -> Settings:
    """
    Загрузить конфиг из YAML и провалидировать через Pydantic.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Pydantic проверит, что структура соответствует Settings:
    # data.pdf_path, embeddings.model_name, qdrant.url и т.д.
    return Settings.model_validate(raw)


def get_config() -> Settings:
    """
    Ленивая (lazy) загрузка конфига — читаем YAML один раз при первом вызове.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config
