import argparse
import yaml
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.ensemble_model import EnsembleModel
from scripts.utils import load_checkpoint

def simple_grad_cam(model, x, target_layer, class_idx=None):
    """
    Basic Grad-CAM approach for a convolutional layer.
    """
    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations["value"] = out

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_backward_hook(backward_hook)

    logits, _ = model(x)
    if class_idx is None:
        class_idx = torch.argmax(logits, dim=1).item()
    score = logits[:, class_idx]
    model.zero_grad()
    score.backward()

    act = activations["value"]
    grad = gradients["value"]

    fh.remove()
    bh.remove()

    weights = torch.mean(grad, dim=[2, 3], keepdim=True)
    cam = (weights * act).sum(dim=1).squeeze()
    cam = F.relu(cam)
    cam = cam.detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

def overlay_cam_on_image(img_np, cam):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    out = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)
    return out

def main(config_path, checkpoint, image_path, output_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnsembleModel(
        backbones=config["model"]["backbones"],
        gating_network=config["model"]["gating_network"],
        gating_mode=config["model"]["gating_mode"],
        feature_dim=config["model"]["feature_dim"],
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters())  # dummy optimizer
    load_checkpoint(model, optimizer, checkpoint)
    model.eval()

    transform = A.Compose([
        A.Resize(config["data"]["image_size"], config["data"]["image_size"]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil)
    inp = transform(image=img_np)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model(inp)
        pred_class = torch.argmax(logits, dim=1).item()

    # Example: pick the first backbone if it's CNN-based and find its last conv layer
    backbone_name = config["model"]["backbones"][0]
    backbone = model.backbone_modules[backbone_name]
    target_layer = None
    for m in backbone.modules():
        if isinstance(m, torch.nn.Conv2d):
            target_layer = m
    if target_layer is None:
        print("No Conv2d layer found for Grad-CAM in the first backbone.")
        return

    cam = simple_grad_cam(model, inp, target_layer, class_idx=pred_class)
    cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    overlaid = overlay_cam_on_image(img_np, cam_resized)
    Image.fromarray(overlaid).save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--output_path", default="gradcam_output.jpg")
    args = parser.parse_args()

    main(args.config, args.checkpoint, args.image_path, args.output_path)
