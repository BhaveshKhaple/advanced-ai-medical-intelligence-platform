FROM python:3.10-slim

WORKDIR /app

# Install PyTorch CPU
RUN pip install --no-cache-dir \
    torch torchvision \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install app dependencies
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    sqlalchemy \
    opencv-python-headless \
    Pillow \
    numpy \
    google-genai \
    streamlit \
    requests

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
