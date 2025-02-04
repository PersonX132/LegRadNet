import argparse
import yaml
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.ensemble_model import EnsembleModel
from scripts.utils import load_checkpoint

def main(config_path, checkpoint_path, image_path):
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

    optimizer = torch.optim.AdamW(model.parameters())
    start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    model.eval()

    transform = A.Compose([
        A.Resize(config["data"]["image_size"], config["data"]["image_size"]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil)
    image_t = transform(image=img_np)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits, gating_scores = model(image_t)
        probs = torch.softmax(logits, dim=1)[0]
        pred_class = torch.argmax(probs).item()

    print(f"[INFO] Loaded checkpoint from epoch {start_epoch-1}")
    print(f"Predicted class: {pred_class}")
    print(f"Probabilities: {probs.cpu().numpy()}")
    if gating_scores is not None:
        print(f"Gating scores: {gating_scores.cpu().numpy()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image_path", required=True)
    args = parser.parse_args()

    main(args.config, args.checkpoint, args.image_path)
