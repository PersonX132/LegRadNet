import argparse
import yaml
import optuna
import copy
import torch

from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from scripts.data_loader import create_dataloaders
from scripts.train import train_one_epoch, validate_one_epoch
from models.ensemble_model import EnsembleModel

def objective(trial, base_config, device):
    config = copy.deepcopy(base_config)
    # Sample LR, WD from the param_spaces
    lr_low = config["hyperparameter_tuning"]["param_spaces"]["learning_rate"]["low"]
    lr_high = config["hyperparameter_tuning"]["param_spaces"]["learning_rate"]["high"]
    wd_low = config["hyperparameter_tuning"]["param_spaces"]["weight_decay"]["low"]
    wd_high = config["hyperparameter_tuning"]["param_spaces"]["weight_decay"]["high"]

    lr = trial.suggest_float("learning_rate", lr_low, lr_high, log=True)
    wd = trial.suggest_float("weight_decay", wd_low, wd_high, log=True)

    config["training"]["learning_rate"] = lr
    config["training"]["weight_decay"] = wd

    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        csv_path=config["data"]["csv_path"],
        img_dir=config["data"]["dataset_path"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        image_size=config["data"]["image_size"],
        train_val_split=config["data"]["train_val_split"]
    )

    # Build model
    model = EnsembleModel(
        backbones=config["model"]["backbones"],
        gating_network=config["model"]["gating_network"],
        gating_mode=config["model"]["gating_mode"],
        feature_dim=config["model"]["feature_dim"],
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)  # fixed to AdamW
    scheduler = CosineAnnealingLR(optimizer, T_max=3)  # short schedule for quick trials

    epochs = 10 #adjust, based on your time constraint
    best_val_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc, _ = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _ = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    trial.report(best_val_acc, epoch)

    return best_val_acc

def main(config_path):
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not base_config["hyperparameter_tuning"]["enable_optuna"]:
        print("Optuna is disabled in config. Exiting.")
        return

    n_trials = base_config["hyperparameter_tuning"]["n_trials"]

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, base_config, device), n_trials=n_trials)

    print("[INFO] Hyperparameter tuning complete.")
    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the config YAML.")
    args = parser.parse_args()

    main(args.config)
