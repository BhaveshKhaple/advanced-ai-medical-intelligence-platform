FROM python:3.10-slim

WORKDIR /app

# PyTorch CPU
RUN pip install --no-cache-dir torch torchvision \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    opencv-python-headless \
    Pillow \
    numpy \
    sqlalchemy \
    google-genai \
    streamlit

COPY . .

# HF Spaces require the app to listen on port 7860
EXPOSE 7860

# Writable dirs for the runtime user
ENV HOME=/app
RUN mkdir -p /app/gradcam_out && chmod -R 777 /app

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
