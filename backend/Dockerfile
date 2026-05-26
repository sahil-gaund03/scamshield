# Multi-stage build for optimized image size
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies into a wheels directory
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Copy installed dependencies
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Install libgomp1 (required for LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY ml/ /app/ml/
COPY backend/ /app/backend/
COPY datasets/ /app/datasets/
COPY reports/ /app/reports/

# Set Python Path
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
