import random
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class WatermarkDataset(Dataset):
    def __init__(self, images_dir, labels_dir,size=(256, 256), augment=False):

        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.size = size
        self.augment = augment

        self.image_files = sorted([
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"] ])

        print(f"Loaded {len(self.image_files)} images from {self.images_dir}")

    def __len__(self):
        return len(self.image_files)

    # -----------------------------
    # YOLO TXT → MASK (IMPROVED)
    # -----------------------------
    def _yolo_to_mask(self, label_path, width, height):
        mask = np.zeros((height, width), dtype=np.uint8)

        if not label_path.exists():
            return mask

        with open(label_path, "r") as f:
            lines = f.read().splitlines()

        for line in lines:
            cls, xc, yc, w, h = map(float, line.split())

            xc *= width
            yc *= height
            w  *= width
            h  *= height

            x1 = int(xc - w / 2)
            y1 = int(yc - h / 2)
            x2 = int(xc + w / 2)
            y2 = int(yc + h / 2)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)

            mask[y1:y2+1, x1:x2+1] = 1

        # 🔥 SHRINK MASK (VERY IMPORTANT)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

        return mask

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.labels_dir / (img_path.stem + ".txt")

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        mask_np = self._yolo_to_mask(label_path, w, h)
        mask = Image.fromarray(mask_np * 255)

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        if self.augment and random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        if self.augment and random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        img_t = TF.to_tensor(img)
        mask_t = TF.to_tensor(mask)
        mask_t = (mask_t > 0.5).float()

        return img_t, mask_t