# RAG-сервис на основе книг по психологии с FastAPI, LangChain, LangGraph и Qdrant
  
Задача: Обернуть твой RaG в сервис. Использовать следующие инструменты:
fastapi, qdrant, chonkie, langchain, docker (docker compose)
  
### Структура проекта:
first_rag/
├── notebooks/ # Тетрадки
├── тесты
│ └── разные тесты
├── src/ # Основной код
│ ├── __init__.py
│ ├── app.py # FastAPI приложение
│ ├── config.py # Конфигурация (API ключи, пути, Qdrant URL)
│ ├── rag_graph.py # LangGraph: граф для RAG (чанк + эмбед + retrieve + generate)
│ ├── data_loader.py # Загрузка и чанкинг PDF (из вашего ноутбука)
│ ├── embeddings.py # Эмбеддинги (HuggingFace)
│ ├── vector_store.py # Интеграция с Qdrant
│ └── llm.py # Интеграция с LLM (Qwen3-8B через LangChain)
├── data/ # Данные

├── config/ # Конфиг-файлы
│ └── config.yaml # YAML-конфиг (пути, модели и т.д.)
├── requirements.txt # Зависимости
├── Dockerfile # Docker для приложения
├── docker-compose.yml # Compose: app + qdrant
├── .gitignore # Игнор (venv, pyc, etc.)
└── README.md # Описание проекта

## Чтобы проверять тесты пиши 
```
pytest src/tests/ -v -s
```
или конкретный
```
pytest src/tests/test_data_loader.py -v -s
```
s - для вывода принта

## Визулизация графа

[question] ──→ retrieve ──→ [context] ──→ generate ──→ [response]
                                       ↑
                              llm + prompt

