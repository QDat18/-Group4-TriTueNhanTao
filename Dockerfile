# ==========================================
# Dockerfile for deploying FastAPI to Hugging Face Spaces
# ==========================================

# 1. Base Image
FROM python:3.10-slim

# 2. Set Environment Variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# 3. Install System Dependencies (Crucial for OpenCV & InsightFace C++ build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    cmake \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Set Working Directory
WORKDIR /app

# 5. Upgrade pip, setuptools, wheel and pre-install numpy & cython (essential for compiling insightface)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy cython

# 6. Install CPU PyTorch (to keep image size small and build fast)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 7. Copy and install rest of Python requirements
COPY requirement.txt /app/
RUN pip install --no-cache-dir -r requirement.txt

# 8. Copy all project files
COPY . /app/

# 9. Expose Hugging Face Space default port
EXPOSE 7860

# 10. Start Uvicorn ASGI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
