import os
import csv
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class KneeDataset(Dataset):
    """
    Loads image filenames & labels from a CSV, with optional transforms.
    """
    def __init__(self, csv_path, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row["filename"]
                label = int(row["label"])
                self.samples.append((fname, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_file, label = self.samples[idx]
        full_path = os.path.join(self.img_dir, image_file)
        image = Image.open(full_path).convert("RGB")
        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label

def get_train_transforms(image_size=512):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.RandomResizedCrop(image_size, image_size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_val_transforms(image_size=512):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.CenterCrop(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def create_dataloaders(csv_path,
                       img_dir,
                       batch_size,
                       num_workers,
                       image_size=512,
                       train_val_split=0.8):
    dataset_full = KneeDataset(csv_path, img_dir, transform=None)
    full_len = len(dataset_full)
    train_len = int(train_val_split * full_len)
    val_len = full_len - train_len

    indices = list(range(full_len))
    random.shuffle(indices)
    train_indices = indices[:train_len]
    val_indices = indices[train_len:]

    train_subset = torch.utils.data.Subset(dataset_full, train_indices)
    val_subset   = torch.utils.data.Subset(dataset_full, val_indices)

    train_subset.dataset.transform = get_train_transforms(image_size)
    val_subset.dataset.transform   = get_val_transforms(image_size)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
