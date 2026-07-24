import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import os
import uuid

from .predict import load_model, transform

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
os.makedirs(STATIC_DIR, exist_ok=True)


def generate_gradcam(image_path: str) -> str:
    model = load_model()

    # Hook target: last conv layer of EfficientNet-B0
    target_layer = model.features[-1]

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    tensor.requires_grad_(True)

    outputs = model(tensor)
    pred_class = outputs.argmax(dim=1).item()

    model.zero_grad()
    outputs[0, pred_class].backward()

    fh.remove()
    bh.remove()

    grads = gradients[0]              # (1, C, H, W)
    acts = activations[0]             # (1, C, H, W)

    weights = grads.mean(dim=[2, 3], keepdim=True)
    cam = (weights * acts).sum(dim=1).squeeze()
    cam = F.relu(cam)
    cam = cam.detach().numpy()

    # Normalize
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = np.uint8(255 * cam)
    cam = cv2.resize(cam, (224, 224))

    # Overlay on original
    original = cv2.imread(image_path)
    original = cv2.resize(original, (224, 224))
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    out_filename = f"gradcam_{uuid.uuid4().hex[:8]}.png"
    out_path = os.path.join(STATIC_DIR, out_filename)
    cv2.imwrite(out_path, overlay)

    return out_path
