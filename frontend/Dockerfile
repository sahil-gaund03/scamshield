FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend and ml components
COPY frontend/ /app/frontend/
COPY ml/ /app/ml/
COPY reports/ /app/reports/

# Set Python Path
ENV PYTHONPATH=/app

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
