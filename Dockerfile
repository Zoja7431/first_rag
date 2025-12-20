# Dockerfile

# Используем официальный Python образ
FROM python:3.12-slim as builder

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем только requirements
COPY requirements_docker.txt .

RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements_docker.txt

# Копируем только нужный код
COPY src ./src
COPY config ./config
# Если нужны данные PDF – монтируем через volume, не копируем
# COPY data/psichology_books ./data/psichology_books  # обычно не нужно

EXPOSE 8000


# Health check для Docker
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Команда для запуска приложения
CMD ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]