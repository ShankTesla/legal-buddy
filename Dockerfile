# Stage 1: Build stage
FROM python:3.12-slim AS builder

WORKDIR /app

# Set environment variables to optimize Python performance
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install build dependencies (if needed for certain libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies into a temporary directory
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Final runtime stage
FROM python:3.12-slim
WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /install /usr/local

# Copy your source code and data
# This copies the contents of your local ./src into /app/src in the container
COPY ./src ./src
COPY ./data ./data

# Set PYTHONPATH so Python can find your modules inside /src
ENV PYTHONPATH=/app/src

# Expose FastAPI port
EXPOSE 8000

# Command to run the application using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
