# 🫁 Advanced AI Medical Intelligence Platform

End-to-end medical image analysis platform with Deep Learning, Explainable AI (Grad-CAM), and LLM-generated medical reports.

## Architecture

```
Streamlit UI → FastAPI REST API → EfficientNet-B0 + Grad-CAM + Gemini LLM → SQLite DB
```

## Features

- **Deep Learning** — EfficientNet-B0 fine-tuned on 5,863 chest X-rays (Normal vs Pneumonia)
- **Explainable AI** — Grad-CAM heatmap highlighting regions that influenced the prediction
- **LLM Report** — Gemini 1.5 Flash generates a structured radiological report
- **REST API** — FastAPI with auto-generated Swagger docs at `/docs`
- **Database** — SQLite + SQLAlchemy stores full prediction history
- **Docker** — Docker Compose runs API + frontend together

## Quickstart

### 1. Train the model (Google Colab — GPU required)

Open `train/train_colab.ipynb` in Google Colab, enable GPU, and run all cells.  
Download `model.pth` and place it in the `weights/` directory.

### 2. Set environment variables

```bash
export GEMINI_API_KEY=your_key_here
```

### 3. Run with Docker

```bash
docker-compose up --build
```

- **Frontend**: http://localhost:8501  
- **API Docs**: http://localhost:8000/docs

### 4. Run locally (without Docker)

```bash
pip install -r requirements.txt

# Terminal 1 — API
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Frontend
streamlit run streamlit_app.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Upload X-ray → returns diagnosis, confidence, Grad-CAM URL, LLM report |
| `GET` | `/history` | List last 50 predictions |
| `GET` | `/report/{id}` | Get LLM report for a prediction |
| `GET` | `/image/{id}` | Get Grad-CAM image for a prediction |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Model | PyTorch · EfficientNet-B0 |
| XAI | Grad-CAM (manual implementation) |
| LLM | Google Gemini 1.5 Flash |
| API | FastAPI · Uvicorn |
| Database | SQLite · SQLAlchemy |
| Frontend | Streamlit |
| Deployment | Docker · Docker Compose · Hugging Face Spaces |

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — Kaggle  
5,863 images · 2 classes (NORMAL / PNEUMONIA)

## Results

| Metric | Value |
|--------|-------|
| Val Accuracy | ~95%+ |
| Test Accuracy | ~92%+ |
| Model Size | ~20MB |

## Author

**Bhavesh Khaple** — B.Tech AI & Data Science, MIT Aurangabad 2026  
[GitHub](https://github.com/BhaveshKhaple)
