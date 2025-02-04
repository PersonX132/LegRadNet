import os
import torch
from sklearn.metrics import accuracy_score, f1_score

def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, path)

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"] + 1

def compute_metrics(outputs, targets):
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    targets_np = targets.cpu().numpy()
    acc = accuracy_score(targets_np, preds)
    f1 = f1_score(targets_np, preds, average="macro")
    return acc, f1
