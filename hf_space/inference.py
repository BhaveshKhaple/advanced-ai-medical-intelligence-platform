import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import os
import uuid

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pth")
CLASSES = ["NORMAL", "PNEUMONIA"]
STATIC_DIR = os.path.join(os.path.dirname(__file__), "gradcam_out")
os.makedirs(STATIC_DIR, exist_ok=True)

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
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    _model = model
    return model


def predict_image(image_path: str) -> dict:
    model = load_model()
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
        pred_idx = torch.argmax(probs).item()
    return {
        "diagnosis": CLASSES[pred_idx],
        "confidence": round(probs[pred_idx].item() * 100, 2),
        "probabilities": {
            "NORMAL": round(probs[0].item() * 100, 2),
            "PNEUMONIA": round(probs[1].item() * 100, 2),
        },
    }


def generate_gradcam(image_path: str) -> str:
    model = load_model()
    target_layer = model.features[-1]
    activations, gradients = [], []

    def fwd(m, i, o):
        activations.append(o)

    def bwd(m, gi, go):
        gradients.append(go[0])

    fh = target_layer.register_forward_hook(fwd)
    bh = target_layer.register_full_backward_hook(bwd)

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    tensor.requires_grad_(True)

    outputs = model(tensor)
    pred_class = outputs.argmax(dim=1).item()
    model.zero_grad()
    outputs[0, pred_class].backward()
    fh.remove()
    bh.remove()

    grads = gradients[0]
    acts = activations[0]
    weights = grads.mean(dim=[2, 3], keepdim=True)
    cam = F.relu((weights * acts).sum(dim=1).squeeze()).detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(np.uint8(255 * cam), (224, 224))

    original = cv2.resize(cv2.imread(image_path), (224, 224))
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    out_path = os.path.join(STATIC_DIR, f"gradcam_{uuid.uuid4().hex[:8]}.png")
    cv2.imwrite(out_path, overlay)
    return out_path
