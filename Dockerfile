# Dockerfile

# Используем официальный Python образ
FROM python:3.12-slim as builder

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем только requirements
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Копируем только нужный код
COPY src ./src
COPY config ./config
COPY config.yaml ./config.yaml  # если нужен в корне
COPY README.md ./README.md

# Если нужны данные PDF – монтируем через volume, не копируем
# COPY data/psichology_books ./data/psichology_books  # обычно не нужно

EXPOSE 8000


# Health check для Docker
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Команда для запуска приложения
CMD ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]