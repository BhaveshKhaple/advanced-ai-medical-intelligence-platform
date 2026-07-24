import os
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from .database import get_db, init_db, Prediction
from .predict import predict_image
from .gradcam import generate_gradcam
from .llm import generate_report

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)

app = FastAPI(
    title="Advanced AI Medical Intelligence Platform",
    description="End-to-end medical image analysis with Deep Learning, Grad-CAM, and LLM report generation.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
def startup():
    init_db()


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Advanced AI Medical Intelligence Platform is running."}


@app.post("/predict", tags=["Inference"])
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted.")

    # Save uploaded file
    ext = os.path.splitext(file.filename)[-1] or ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, filename)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Inference
    result = predict_image(save_path)

    # Grad-CAM
    gradcam_path = generate_gradcam(save_path)
    gradcam_filename = os.path.basename(gradcam_path)

    # LLM Report
    report = generate_report(
        diagnosis=result["diagnosis"],
        confidence=result["confidence"],
        normal_prob=result["probabilities"]["NORMAL"],
        pneumonia_prob=result["probabilities"]["PNEUMONIA"],
    )

    # Store in DB
    prediction = Prediction(
        image_filename=file.filename,
        image_path=save_path,
        gradcam_path=gradcam_path,
        diagnosis=result["diagnosis"],
        confidence=result["confidence"],
        llm_report=report,
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    return JSONResponse({
        "id": prediction.id,
        "diagnosis": result["diagnosis"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "gradcam_url": f"/static/{gradcam_filename}",
        "report": report,
        "created_at": str(prediction.created_at),
    })


@app.get("/history", tags=["History"])
def get_history(limit: int = 50, db: Session = Depends(get_db)):
    records = (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "image_filename": r.image_filename,
            "diagnosis": r.diagnosis,
            "confidence": r.confidence,
            "gradcam_url": f"/static/{os.path.basename(r.gradcam_path)}" if r.gradcam_path else None,
            "created_at": str(r.created_at),
        }
        for r in records
    ]


@app.get("/report/{prediction_id}", tags=["History"])
def get_report(prediction_id: int, db: Session = Depends(get_db)):
    record = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    return {
        "id": record.id,
        "diagnosis": record.diagnosis,
        "confidence": record.confidence,
        "llm_report": record.llm_report,
        "created_at": str(record.created_at),
    }


@app.get("/image/{prediction_id}", tags=["History"])
def get_image(prediction_id: int, db: Session = Depends(get_db)):
    record = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not record or not record.gradcam_path:
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(record.gradcam_path, media_type="image/png")
