"""
vector_store.py - Интеграция с Qdrant векторной базой данных

Управление сохранением и поиском эмбеддингов в Qdrant
"""

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.config import get_config
from src.embeddings import get_embeddings_model

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Менеджер для работы с Qdrant векторной базой"""

    _instance: Optional["VectorStoreManager"] = None
    _vector_store: Optional[QdrantVectorStore] = None
    _client: Optional[QdrantClient] = None

    def __new__(cls):
        """Singleton паттерн"""
        if cls._instance is None:
            cls._instance = super(VectorStoreManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Инициализация менеджера векторной базы"""
        if self._client is None:
            self._init_client()
        if self._vector_store is None:
            self._init_vector_store()

    def _init_client(self) -> None:
        """Инициализация Qdrant клиента"""
        config = get_config()
        qdrant_config = config.qdrant

        logger.info(f"🔗 Connecting to Qdrant at {qdrant_config.full_url}")

        try:
            # Убедиться что строки не содержат невалидных символов
            url = str(qdrant_config.full_url).strip()
            api_key = str(qdrant_config.api_key).strip()

            self._client = QdrantClient(url=url, api_key=api_key, timeout=30)

            # Проверяем соединение
            info = self._client.get_collections()
            logger.info(f"✅ Connected to Qdrant")
            logger.info(f"   URL: {url[:50]}...")
            logger.info(f"   Collections: {len(info.collections)}")

        except UnicodeEncodeError as e:
            logger.error(
                f"⚠️  Ошибка кодирования (проверьте .env и config.yaml): {e}"
            )
            logger.error(f"   API Key format: {repr(qdrant_config.api_key[:20])}...")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to connect to Qdrant: {e}")
            raise

    def _init_vector_store(self) -> None:
        """Инициализация или загрузка векторной базы"""
        config = get_config()
        qdrant_config = config.qdrant
        embeddings = get_embeddings_model()

        logger.info(f"🏗️  Initializing vector store: {qdrant_config.collection_name}")

        try:
            # 1. Проверяем, существует ли коллекция
            try:
                collection_info = self._client.get_collection(
                    collection_name=qdrant_config.collection_name
                )
                logger.info(f"✅ Collection '{qdrant_config.collection_name}' exists")
                logger.info(f"   Points: {collection_info.points_count}")

            except Exception:
                logger.info(
                    f"📝 Collection '{qdrant_config.collection_name}' not found, creating..."
                )

                # 2. Создаём коллекцию с нужным размером вектора
                self._client.create_collection(
                    collection_name=qdrant_config.collection_name,
                    vectors_config=VectorParams(
                        size=qdrant_config.vector_size,
                        distance=Distance.COSINE,
                    ),
                )

                logger.info(f"✅ Collection '{qdrant_config.collection_name}' created")

            # 3. Инициализируем QdrantVectorStore, используя уже созданный client
            self._vector_store = QdrantVectorStore(
                client=self._client,
                collection_name=qdrant_config.collection_name,
                embedding=embeddings,
                distance=Distance.COSINE,
            )

            logger.info("✅ Vector store initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize vector store: {e}")
            raise

    def get_vector_store(self) -> QdrantVectorStore:
        """
        Получить объект векторной базы

        Returns:
            QdrantVectorStore: Объект для работы с векторной базой
        """
        if self._vector_store is None:
            self._init_vector_store()
        return self._vector_store

    def get_client(self) -> QdrantClient:
        """
        Получить Qdrant клиент

        Returns:
            QdrantClient: Qdrant клиент
        """
        if self._client is None:
            self._init_client()
        return self._client

    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        Добавить документы в векторную базу

        Args:
            documents: Список документов для добавления

        Returns:
            list[str]: Список ID добавленных документов
        """
        if not documents:
            logger.warning("⚠️  Empty documents list provided")
            return []

        logger.info(f"📥 Adding {len(documents)} documents to vector store...")

        try:
            # Добавляем документы через LangChain интерфейс
            ids = self._vector_store.add_documents(documents)
            logger.info(f"✅ Added {len(ids)} documents successfully")
            return ids

        except Exception as e:
            logger.error(f"❌ Error adding documents: {e}")
            raise

    def search(self, query: str, k: int = 4) -> list[Document]:
        """
        Поиск похожих документов по запросу

        Args:
            query: Текст запроса
            k: Количество результатов для возврата

        Returns:
            list[Document]: Список найденных документов, отсортированный по релевантности
        """
        config = get_config()
        k = k or config.qdrant.search_limit

        logger.debug(f"🔍 Searching for: {query[:100]}...")

        try:
            # Выполняем поиск через similarity_search (использует косинусное расстояние)
            results = self._vector_store.similarity_search(query=query, k=k)
            logger.debug(f"   Found {len(results)} similar documents")
            return results

        except Exception as e:
            logger.error(f"❌ Error searching documents: {e}")
            raise

    def search_with_scores(
        self, query: str, k: int = 4
    ) -> list[tuple[Document, float]]:
        """
        Поиск похожих документов с оценками релевантности

        Args:
            query: Текст запроса
            k: Количество результатов для возврата

        Returns:
            list[tuple]: Список кортежей (Document, score)
        """
        config = get_config()
        k = k or config.qdrant.search_limit

        try:
            results = self._vector_store.similarity_search_with_score(query=query, k=k)
            logger.debug(f"🔍 Found {len(results)} similar documents with scores")
            return results

        except Exception as e:
            logger.error(f"❌ Error searching documents with scores: {e}")
            raise

    def get_collection_stats(self) -> dict:
        """
        Получить статистику коллекции

        Returns:
            dict: Информация о коллекции
        """
        config = get_config()

        try:
            collection_info = self._client.get_collection(
                collection_name=config.qdrant.collection_name
            )

            return {
                "collection_name": config.qdrant.collection_name,
                "points_count": collection_info.points_count,
                "vectors_count": getattr(
                    collection_info, "vectors_count", collection_info.points_count
                ),
                "segments_count": getattr(collection_info, "segments_count", None),
            }

        except Exception as e:
            logger.error(f"❌ Error getting collection stats: {e}")
            raise

    def delete_collection(self) -> bool:
        """
        Удалить коллекцию (осторожно!)

        Returns:
            bool: True если успешно удалена
        """
        config = get_config()
        collection_name = config.qdrant.collection_name

        try:
            self._client.delete_collection(collection_name=collection_name)
            logger.warning(f"🗑️  Collection '{collection_name}' deleted")
            self._vector_store = None  # Сбросим ссылку

            return True

        except Exception as e:
            logger.error(f"❌ Error deleting collection: {e}")
            return False


def get_vector_store_manager() -> VectorStoreManager:
    """
    Получить синглтон менеджер векторной базы

    Returns:
        VectorStoreManager: Менеджер векторной базы
    """
    return VectorStoreManager()


def get_vector_store() -> QdrantVectorStore:
    """
    Получить объект векторной базы (для совместимости)

    Returns:
        QdrantVectorStore: Объект векторной базы
    """
    return get_vector_store_manager().get_vector_store()


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Тест управления векторной базой
    manager = get_vector_store_manager()

    # Получаем статистику
    stats = manager.get_collection_stats()
    print(f"\n📊 Collection Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Тест поиска (если есть документы)
    if stats["points_count"] > 0:
        test_query = "психология поведения"
        results = manager.search(test_query, k=2)
        print(f"\n🔍 Search results for: '{test_query}'")
        for i, doc in enumerate(results, 1):
            print(f"   {i}. Source: {doc.metadata.get('source')}")
            print(f"      Text: {doc.page_content[:100]}...")