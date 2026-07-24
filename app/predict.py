import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "weights", "model.pth")
CLASSES = ["NORMAL", "PNEUMONIA"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_model = None


def load_model():
    global _model
    if _model is not None:
        return _model

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    else:
        raise FileNotFoundError(
            f"Model weights not found at {MODEL_PATH}. "
            "Please run train/train.ipynb on Colab first and download model.pth."
        )

    model.eval()
    _model = model
    return model


def predict_image(image_path: str) -> dict:
    model = load_model()
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    return {
        "diagnosis": CLASSES[pred_idx],
        "confidence": round(probs[pred_idx].item() * 100, 2),
        "probabilities": {
            "NORMAL": round(probs[0].item() * 100, 2),
            "PNEUMONIA": round(probs[1].item() * 100, 2),
        },
    }
