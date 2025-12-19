"""
config.py — загрузка конфигурации из config.yaml без дублирования значений.

Все конкретные значения (пути, model_name, url, api_key и т.д.)
берутся ТОЛЬКО из config/config.yaml.
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Ищет .env в корне проекта
load_dotenv(dotenv_path=".env", override=True)

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

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Убедиться что api_key содержит только ASCII символы"""
        if not isinstance(v, str):
            raise ValueError("api_key должен быть строкой")
        if not v or v.startswith("$"):
            raise ValueError(
                f"api_key не загружен из .env или содержит подстановку: {v}"
            )
        try:
            v.encode("ascii")
        except UnicodeEncodeError:
            raise ValueError(
                f"api_key содержит недопустимые символы (кириллица?): {v}"
            )
        return v.strip()

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Убедиться что url содержит только ASCII символы"""
        if not isinstance(v, str):
            raise ValueError("url должен быть строкой")
        if not v or v.startswith("$"):
            raise ValueError(
                f"url не загружен из .env или содержит подстановку: {v}"
            )
        try:
            v.encode("ascii")
        except UnicodeEncodeError:
            raise ValueError(f"url содержит недопустимые символы (кириллица?): {v}")
        return v.strip()

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


class HuggingFaceLLM(BaseModel):
    model_id: str
    api_key: str
    temperature: float = 0.1
    max_new_tokens: int = 1024

class GroqLLM(BaseModel):
    api_key: str = Field(..., description="Groq API key из .env")
    models: List[str]
    temperature: float = Field(default=0.1, description="Температура генерации")
    max_tokens: int = Field(default=1024, description="Макс токенов")

    @field_validator("api_key")
    @classmethod
    def validate_groq_api_key(cls, v: str) -> str:
        """Убедиться что api_key из .env загружен корректно"""
        if not isinstance(v, str):
            raise ValueError("Groq api_key должен быть строкой")
        if not v or v.startswith("$"):
            raise ValueError(
                f"Groq api_key не загружен из .env или содержит подстановку: {v}"
            )
        try:
            v.encode("ascii")
        except UnicodeEncodeError:
            raise ValueError(
                f"Groq api_key содержит недопустимые символы: {v}"
            )
        return v.strip()
    
class LLMConfig(BaseModel):
    type: str = Field(..., description="Тип LLM: huggingface | groq")
    huggingface: Optional[HuggingFaceLLM] = None
    groq: Optional[GroqLLM] = None

class Settings(BaseModel):
    data: DataConfig
    text_processing: TextProcessingConfig
    embeddings: EmbeddingsConfig
    qdrant: QdrantConfig
    prompt: PromptConfig
    llm: LLMConfig = Field(..., description="LLM конфигурация")

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
    
    Поддерживает переменные окружения в формате ${VAR_NAME} или $VAR_NAME
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Подставляем переменные окружения (поддержка ${VAR} и $VAR)
    raw = _substitute_env_vars(raw)

    logger.info(f"✅ Loaded config from {path}")

    # Pydantic проверит структуру и валидность
    return Settings.model_validate(raw)


def _substitute_env_vars(obj: Any) -> Any:
    """
    Рекурсивно подставляет переменные окружения в конфиг.
    Поддержка: ${VAR_NAME}, $VAR_NAME
    """
    import re

    if isinstance(obj, dict):
        return {key: _substitute_env_vars(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [_substitute_env_vars(item) for item in obj]

    if isinstance(obj, str):
        # Паттерн ${VAR_NAME} или $VAR_NAME
        def replace_var(match):
            var_name = match.group(1) or match.group(2)
            value = os.getenv(var_name)
            if value is None:
                logger.warning(f"⚠️  Environment variable {var_name} not found!")
                return match.group(0)  # Возвращаем оригинальную строку если не найдена
            return value

        # Поддерживаем ${VAR} и $VAR
        obj = re.sub(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)", replace_var, obj)

    return obj


def get_config() -> Settings:
    """
    Ленивая (lazy) загрузка конфига — читаем YAML один раз при первом вызове.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config() -> Settings:
    """
    Перезагрузить конфиг (полезно для тестов).
    """
    global _config
    _config = load_config()
    return _config
