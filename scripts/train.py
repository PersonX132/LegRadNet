import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from scripts.data_loader import create_dataloaders
from scripts.utils import save_checkpoint, compute_metrics
from models.ensemble_model import EnsembleModel

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, total_acc, total_f1 = 0, 0, 0
    count = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        acc, f1 = compute_metrics(logits, labels)
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_acc += acc * bs
        total_f1 += f1 * bs
        count += bs

    return total_loss / count, total_acc / count, total_f1 / count

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_acc, total_f1 = 0, 0, 0
    count = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits, _ = model(images)
            loss = criterion(logits, labels)
            acc, f1 = compute_metrics(logits, labels)
            bs = images.size(0)
            total_loss += loss.item() * bs
            total_acc += acc * bs
            total_f1 += f1 * bs
            count += bs

    return total_loss / count, total_acc / count, total_f1 / count

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = create_dataloaders(
        csv_path=config["data"]["csv_path"],
        img_dir=config["data"]["dataset_path"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        image_size=config["data"]["image_size"],
        train_val_split=config["data"]["train_val_split"]
    )

    model = EnsembleModel(
        backbones=config["model"]["backbones"],
        gating_network=config["model"]["gating_network"],
        gating_mode=config["model"]["gating_mode"],
        feature_dim=config["model"]["feature_dim"],
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    if config["training"]["optimizer"] == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"]
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            momentum=0.9
        )

    scheduler = None
    if config["training"]["lr_scheduler"] == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])

    epochs = config["training"]["epochs"]
    best_val_acc = 0.0

    for epoch in range(epochs):
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = validate_one_epoch(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val   F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, config["training"]["checkpoint_dir"])

    print("Training finished. Best val_acc:", best_val_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/ensemble_config.yaml")
    args = parser.parse_args()

    main(args.config)
