FROM python:3.11-slim

# Install Tesseract OCR (fallback) and OpenCV system deps
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (ONNX Runtime, not PyTorch — keeps image ~500MB vs ~3GB)
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy application code and model
COPY main.py .
COPY app/ app/
COPY templates/ templates/
COPY static/ static/
COPY sw.js .
COPY Examples/ Examples/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
