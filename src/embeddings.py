"""
embeddings.py - Инициализация и управление эмбеддингами
Использует HuggingFace Sentence Transformers для создания эмбеддингов
"""

import logging
from typing import Optional
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EmbeddingsConfig, get_config

logger = logging.getLogger(__name__)


class EmbeddingsManager:
    """Менеджер для работы с эмбеддингами"""
    
    _instance: Optional['EmbeddingsManager'] = None
    _embeddings_model: Optional[HuggingFaceEmbeddings] = None
    
    def __new__(cls):
        """Singleton паттерн"""
        if cls._instance is None:
            cls._instance = super(EmbeddingsManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Инициализация менеджера эмбеддингов"""
        if self._embeddings_model is None:
            self._load_embeddings()
    
    def _load_embeddings(self) -> None:
        """Загрузка модели эмбеддингов"""
        config = get_config()
        emb_config = config.embeddings
        
        logger.info(f"Loading embeddings model: {emb_config.model_name}")
        
        try:
            self._embeddings_model = HuggingFaceEmbeddings(
                model_name=emb_config.model_name,
                model_kwargs={
                    'device': emb_config.device,
                    **emb_config.model_kwargs
                },
                encode_kwargs={
                    'normalize_embeddings': True
                }
            )
            
            logger.info(f"✅ Embeddings model loaded successfully")
            logger.info(f"   Model: {emb_config.model_name}")
            logger.info(f"   Device: {emb_config.device}")
            logger.info(f"   Embedding dimension: {emb_config.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load embeddings model: {e}")
            raise
    
    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Получить объект эмбеддингов
        
        Returns:
            HuggingFaceEmbeddings: Объект для работы с эмбеддингами
        """
        if self._embeddings_model is None:
            self._load_embeddings()
        return self._embeddings_model
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Создать эмбеддинг для текста
        
        Args:
            text: Текст для эмбеддинга
        
        Returns:
            np.ndarray: Вектор эмбеддинга размерности embedding_dim
        """
        if self._embeddings_model is None:
            self._load_embeddings()
        
        embedding = self._embeddings_model.embed_query(text)
        return np.array(embedding)
    
    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        """
        Создать эмбеддинги для списка текстов
        
        Args:
            texts: Список текстов
        
        Returns:
            list[np.ndarray]: Список векторов эмбеддингов
        """
        if self._embeddings_model is None:
            self._load_embeddings()
        
        embeddings = self._embeddings_model.embed_documents(texts)
        return [np.array(emb) for emb in embeddings]
    
    def get_embedding_dim(self) -> int:
        """
        Получить размерность эмбеддинга
        
        Returns:
            int: Размерность вектора эмбеддинга
        """
        config = get_config()
        return config.embeddings.embedding_dim


def get_embeddings_manager() -> EmbeddingsManager:
    """
    Получить синглтон менеджер эмбеддингов
    
    Returns:
        EmbeddingsManager: Менеджер эмбеддингов
    """
    return EmbeddingsManager()


def get_embeddings_model() -> HuggingFaceEmbeddings:
    """
    Получить модель эмбеддингов (для совместимости с LangChain)
    
    Returns:
        HuggingFaceEmbeddings: Модель эмбеддингов
    """
    return get_embeddings_manager().get_embeddings()


if __name__ == "__main__":
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Тест эмбеддингов
    manager = get_embeddings_manager()
    
    # Тест одного текста
    test_text = "Психология - это наука о поведении и психических процессах"
    embedding = manager.embed_text(test_text)
    
    print(f"\n📊 Тест эмбеддингов:")
    print(f"Текст: {test_text}")
    print(f"Размер эмбеддинга: {len(embedding)}")
    print(f"Первые 5 элементов: {embedding[:5]}")
    print(f"Норма вектора: {np.linalg.norm(embedding):.4f}")
    
    # Тест нескольких текстов
    texts = [
        "Психология изучает поведение человека",
        "Эмбеддинги преобразуют текст в числовые векторы"
    ]
    
    embeddings = manager.embed_texts(texts)
    print(f"\n✅ Создано эмбеддингов: {len(embeddings)}")
    
    # Вычисляем косинусное сходство
    similarity = np.dot(embeddings[0], embeddings[1])
    print(f"Косинусное сходство между текстами: {similarity:.4f}")