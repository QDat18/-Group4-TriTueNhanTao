# ==========================================
# Dockerfile for deploying FastAPI to Hugging Face Spaces
# ==========================================

# 1. Base Image
FROM python:3.10-slim

# 2. Set Environment Variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# 3. Install System Dependencies (Crucial for OpenCV & InsightFace)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Set Working Directory
WORKDIR /app

# 5. Install CPU PyTorch first (to keep image size small and build fast)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 6. Copy and install rest of Python requirements
COPY requirement.txt /app/
RUN pip install --no-cache-dir -r requirement.txt

# 7. Copy all project files
COPY . /app/

# 8. Expose Hugging Face Space default port
EXPOSE 7860

# 9. Start Uvicorn ASGI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
