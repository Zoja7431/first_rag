"""
src/app.py — FastAPI приложение для RAG системы психолога

REST API для взаимодействия с RAG пайплайном:

- POST /api/ask — задать вопрос (основной endpoint)
- GET /api/health — проверка здоровья сервиса
- GET /api/stats — статистика Qdrant коллекции
- POST /api/index — индексировать PDF документы
"""

# ═════════════════════════════════════════════════════════════════
# БЛОК 1: ИМПОРТЫ
# ═════════════════════════════════════════════════════════════════

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path  # ✅ ДОБАВЛЕНО

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import get_config
from src.rag_graph import ask as rag_ask
from src.vector_store import get_vector_store_manager
from src.data_loader import get_pdf_chunker  # ✅ ИСПРАВЛЕНО

# ═════════════════════════════════════════════════════════════════
# БЛОК 2: ЛОГИРОВАНИЕ
# ═════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
logger.info("🚀 Инициализация FastAPI приложения...")

# ═════════════════════════════════════════════════════════════════
# БЛОК 3: PYDANTIC МОДЕЛИ
# ═════════════════════════════════════════════════════════════════

class QuestionRequest(BaseModel):
    """Входящий запрос: вопрос для RAG"""
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Вопрос для RAG системы"
    )
    k: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Количество релевантных документов"
    )

    class Config:
        example = {
            "question": "Как читать язык телодвижений?",
            "k": 4
        }


class QuestionResponse(BaseModel):
    """Исходящий ответ на /api/ask"""
    question: str = Field(..., description="Исходный вопрос")
    answer: str = Field(..., description="Ответ от RAG")
    processing_time: float = Field(..., description="Время обработки (сек)")
    documents_count: int = Field(..., description="Сколько документов найдено")

    class Config:
        example = {
            "question": "Как читать язык телодвижений?",
            "answer": "Язык телодвижений, также известный как язык тела...",
            "processing_time": 2.5,
            "documents_count": 4
        }


class HealthResponse(BaseModel):
    """Ответ на /api/health"""
    status: str = Field(..., description="healthy или unhealthy")
    qdrant_connected: bool
    collections_count: int
    config_loaded: bool
    message: str = Field(default="")

    class Config:
        example = {
            "status": "healthy",
            "qdrant_connected": True,
            "collections_count": 1,
            "config_loaded": True,
            "message": "✅ All systems operational"
        }


class StatsResponse(BaseModel):
    """Ответ на /api/stats"""
    collection_name: str
    points_count: int
    vectors_count: int
    segments_count: Optional[int] = None
    embedding_model: str
    embedding_dim: int

    class Config:
        example = {
            "collection_name": "psychology_rag",
            "points_count": 1468,
            "vectors_count": 1468,
            "segments_count": 5,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dim": 384
        }


class IndexResponse(BaseModel):
    """Ответ на /api/index"""
    status: str
    documents_indexed: int
    chunks_created: int
    processing_time: float
    message: str = Field(default="")

    class Config:
        example = {
            "status": "success",
            "documents_indexed": 2,
            "chunks_created": 1468,
            "processing_time": 15.2,
            "message": "✅ Documents indexed successfully"
        }


# ═════════════════════════════════════════════════════════════════
# БЛОК 4: LIFESPAN (STARTUP/SHUTDOWN)
# ═════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Жизненный цикл приложения: startup + shutdown"""

    # ═ STARTUP ═
    logger.info("🚀 Запуск FastAPI сервера...")
    logger.info("📚 Загрузка конфигурации...")

    try:
        config = get_config()
        logger.info(f"✅ Конфиг загружен: коллекция '{config.qdrant.collection_name}'")

        logger.info("🔗 Подключение к Qdrant...")
        vs_manager = get_vector_store_manager()
        stats = vs_manager.get_collection_stats()
        logger.info(f"✅ Qdrant подключен: {stats['points_count']} документов")

    except Exception as e:
        logger.error(f"❌ Ошибка инициализации: {e}")
        raise

    logger.info("✅ Сервер готов к работе!")

    yield  # Сервер работает здесь

    # ═ SHUTDOWN ═
    logger.info("🛑 Остановка FastAPI сервера...")
    logger.info("✅ Сервер остановлен")


# ═════════════════════════════════════════════════════════════════
# БЛОК 5: СОЗДАНИЕ FastAPI ПРИЛОЖЕНИЯ
# ═════════════════════════════════════════════════════════════════

app = FastAPI(
    title="RAG Psychology System API",
    description="REST API для системы психолога на основе RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

logger.info("✅ FastAPI приложение создано")

# ═════════════════════════════════════════════════════════════════
# БЛОК 6: CORS MIDDLEWARE
# ═════════════════════════════════════════════════════════════════

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("✅ CORS middleware добавлен")

# ═════════════════════════════════════════════════════════════════
# БЛОК 7: ROOT ENDPOINT (GET /)
# ═════════════════════════════════════════════════════════════════

@app.get("/", tags=["Root"])
async def root():
    """Главная страница API"""
    return {
        "name": "RAG Psychology System API",
        "version": "1.0.0",
        "description": "REST API для системы психолога на основе RAG",
        "docs": "/docs",
        "docs_redoc": "/redoc",
    }


# ═════════════════════════════════════════════════════════════════
# БЛОК 8: HEALTH CHECK (GET /api/health)
# ═════════════════════════════════════════════════════════════════

@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """🏥 Проверка здоровья сервиса"""

    try:
        config = get_config()
        vs_manager = get_vector_store_manager()
        client = vs_manager.get_client()
        collections = client.get_collections()

        return HealthResponse(
            status="healthy",
            qdrant_connected=True,
            collections_count=len(collections.collections),
            config_loaded=True,
            message="✅ All systems operational",
        )

    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            qdrant_connected=False,
            collections_count=0,
            config_loaded=False,
            message=f"❌ Error: {str(e)}",
        )


# ═════════════════════════════════════════════════════════════════
# БЛОК 9: ГЛАВНЫЙ ENDPOINT (POST /api/ask) - RAG!
# ═════════════════════════════════════════════════════════════════

@app.post("/api/ask", response_model=QuestionResponse, tags=["RAG"])
async def ask_question(request: QuestionRequest) -> QuestionResponse:
    """
    🎯 Задать вопрос RAG системе

    Входные параметры:
    - question: вопрос (1-1000 символов)
    - k: количество документов (1-20, по умолчанию 4)

    Возвращает:
    - question: исходный вопрос
    - answer: ответ от LLM
    - processing_time: время обработки
    - documents_count: сколько документов найдено
    """

    start_time = time.time()

    try:
        logger.info(f"📝 Обработка вопроса: {request.question[:50]}...")

        # Вызываем RAG пайплайн
        answer = rag_ask(request.question)

        # Получаем найденные документы
        vs_manager = get_vector_store_manager()
        docs = vs_manager.search(request.question, k=request.k)

        # Вычисляем время
        processing_time = time.time() - start_time

        logger.info(
            f"✅ Ответ за {processing_time:.2f}s ({len(docs)} документов)"
        )

        return QuestionResponse(
            question=request.question,
            answer=answer,
            processing_time=round(processing_time, 2),
            documents_count=len(docs),
        )

    except Exception as e:
        logger.error(f"❌ Ошибка при обработке: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}",
        )


# ═════════════════════════════════════════════════════════════════
# БЛОК 10: STATS ENDPOINT (GET /api/stats)
# ═════════════════════════════════════════════════════════════════

@app.get("/api/stats", response_model=StatsResponse, tags=["Stats"])
async def get_stats() -> StatsResponse:
    """📊 Получить статистику Qdrant коллекции"""

    try:
        config = get_config()
        vs_manager = get_vector_store_manager()
        stats = vs_manager.get_collection_stats()

        logger.info(f"📊 Статистика запрошена: {stats}")

        return StatsResponse(
            collection_name=stats["collection_name"],
            points_count=stats["points_count"],
            vectors_count=stats.get("vectors_count", stats["points_count"]),
            segments_count=stats.get("segments_count"),
            embedding_model=config.embeddings.model_name,
            embedding_dim=config.embeddings.embedding_dim,
        )

    except Exception as e:
        logger.error(f"❌ Ошибка получения статистики: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting stats: {str(e)}",
        )


# ═════════════════════════════════════════════════════════════════
# БЛОК 11: INDEX ENDPOINT (POST /api/index) ✅ ИСПРАВЛЕНО!
# ═════════════════════════════════════════════════════════════════

@app.post("/api/index", response_model=IndexResponse, tags=["Admin"])
async def index_documents() -> IndexResponse:
    """
    📚 Индексировать PDF документы в Qdrant

    ⚠️ МОЖЕТ БЫТЬ ДОЛГОЙ ОПЕРАЦИЕЙ (15-30 сек)
    """

    try:
        config = get_config()
        logger.info(f"📚 Начало индексирования из {config.data.pdf_path}...")

        start_time = time.time()

        # ✅ ИСПРАВЛЕНО: Используем get_pdf_chunker вместо DataLoader
        chunker = get_pdf_chunker()

        # Получаем все PDF файлы из папки
        pdf_path = Path(config.data.pdf_path)
        pdf_files = list(pdf_path.glob("*.pdf"))

        if not pdf_files:
            raise ValueError(f"Нет PDF файлов в {pdf_path}")

        logger.info(f"📄 Найдено {len(pdf_files)} PDF файлов")

        # ✅ ИСПРАВЛЕНО: Используем load_multiple вместо load_and_process_pdfs
        documents = chunker.load_multiple([str(f) for f in pdf_files])

        logger.info(f"📄 Загружено {len(documents)} документов/чанков")

        # Добавляем в Qdrant
        vs_manager = get_vector_store_manager()
        vs_manager.add_documents(documents)

        processing_time = time.time() - start_time

        logger.info(
            f"✅ Индексирование завершено: {len(documents)} документов "
            f"за {processing_time:.2f}s"
        )

        return IndexResponse(
            status="success",
            documents_indexed=len(pdf_files),
            chunks_created=len(documents),
            processing_time=round(processing_time, 2),
            message=f"✅ {len(documents)} chunks from {len(pdf_files)} documents indexed",
        )

    except Exception as e:
        logger.error(f"❌ Ошибка при индексировании: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error indexing documents: {str(e)}",
        )


# ═════════════════════════════════════════════════════════════════
# БЛОК 12: ERROR HANDLERS
# ═════════════════════════════════════════════════════════════════

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Обработчик ValueError"""
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Обработчик всех остальных исключений"""
    logger.error(f"❌ Необработанное исключение: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ═════════════════════════════════════════════════════════════════
# БЛОК 13: MAIN
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """Запуск как скрипт"""
    import uvicorn

    config = get_config()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )


logger.info("✅ src/app.py загружен полностью")