# Use python:3.11-slim as base image
FROM python:3.11-slim

# Set environment variables to optimize Hugging Face cache and prevent python buffering
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    PORT=8080 \
    SERVER_NAME=0.0.0.0

# Set working directory
WORKDIR /app

# Install system dependencies (build-essential for hnswlib/chromadb compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install CPU-only PyTorch first to reduce image size, then install the rest of requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the Hugging Face embedding model during build time
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5', model_kwargs={'device': 'cpu'})"

# Copy the rest of the application files
COPY . .

# Expose port
EXPOSE 8080

# Start Gradio application
CMD ["python", "app.py"]
