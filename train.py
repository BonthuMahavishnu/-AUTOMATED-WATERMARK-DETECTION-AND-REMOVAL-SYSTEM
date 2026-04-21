import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from model.unet import UNet
from model.dataset import WatermarkDataset
import random
import time
import multiprocessing as mp

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = torch.device("cpu")   # force CPU (faster stability)

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "WatermarkDataset"

IMAGES_DIR = DATA_ROOT / "images"
LABELS_DIR = DATA_ROOT / "labels"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (256,256)
BATCH_SIZE = 6
EPOCHS = 10
LR = 5e-4

# 🔥 TRAIN ONLY SMALL SUBSET
TRAIN_SAMPLES = 4000
VAL_SAMPLES = 1000

# -----------------------------
# LOSS FUNCTIONS
# -----------------------------
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):

        probs = torch.sigmoid(logits)

        probs = probs.view(probs.size(0),-1)
        targets = targets.view(targets.size(0),-1)

        intersection = (probs*targets).sum(dim=1)
        union = probs.sum(dim=1)+targets.sum(dim=1)

        dice = (2*intersection+self.eps)/(union+self.eps)

        return 1-dice.mean()

# -----------------------------
# METRICS
# -----------------------------
def pixel_accuracy(logits,masks,thresh=0.5):

    probs = torch.sigmoid(logits)
    preds = (probs>thresh).float()

    return (preds==masks).sum().item()/masks.numel()


def compute_iou(preds,masks,thresh=0.5):

    preds = torch.sigmoid(preds)
    preds = (preds>thresh).float()

    intersection = (preds*masks).sum()
    union = preds.sum()+masks.sum()-intersection

    if union==0:
        return 1.0

    return (intersection/union).item()

# -----------------------------
# TRAIN
# -----------------------------
def main():

    print("🚀 Training started on:",DEVICE)

    train_ds = WatermarkDataset(
        IMAGES_DIR/"train",
        LABELS_DIR/"train",
        size=IMG_SIZE,
        augment=True
    )

    val_ds = WatermarkDataset(
        IMAGES_DIR/"val",
        LABELS_DIR/"val",
        size=IMG_SIZE,
        augment=False
    )

    # 🔥 RANDOM SUBSET
    train_indices = random.sample(range(len(train_ds)),TRAIN_SAMPLES)
    val_indices = random.sample(range(len(val_ds)),VAL_SAMPLES)

    train_ds = Subset(train_ds,train_indices)
    val_ds = Subset(val_ds,val_indices)

    train_loader = DataLoader(train_ds,BATCH_SIZE,shuffle=True)
    val_loader = DataLoader(val_ds,BATCH_SIZE,shuffle=False)

    model = UNet().to(DEVICE)

    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()

    optimizer = torch.optim.Adam(model.parameters(),lr=LR)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):

        print(f"\n📘 Epoch {epoch+1}/{EPOCHS}")

        model.train()

        train_loss=0
        train_acc=0
        train_iou=0
        batches=0

        for imgs,masks in train_loader:

            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(imgs)

            loss = bce(preds,masks)+dice(preds,masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += pixel_accuracy(preds,masks)
            train_iou += compute_iou(preds,masks)

            batches +=1

        train_loss/=batches
        train_acc/=batches
        train_iou/=batches

        # -----------------------------
        # VALIDATION
        # -----------------------------
        model.eval()

        val_loss=0
        val_acc=0
        val_iou=0
        batches=0

        with torch.no_grad():

            for imgs,masks in val_loader:

                imgs = imgs.to(DEVICE)
                masks = masks.to(DEVICE)

                preds = model(imgs)

                loss = bce(preds,masks)+dice(preds,masks)

                val_loss+=loss.item()
                val_acc+=pixel_accuracy(preds,masks)
                val_iou+=compute_iou(preds,masks)

                batches+=1

        val_loss/=batches
        val_acc/=batches
        val_iou/=batches

        print(
            f"Train Loss:{train_loss:.4f} | Train Acc:{train_acc*100:.2f}% | Train IoU:{train_iou:.3f}"
        )

        print(
            f"Val Loss:{val_loss:.4f} | Val Acc:{val_acc*100:.2f}% | Val IoU:{val_iou:.3f}"
        )

        torch.save(model.state_dict(),CHECKPOINT_DIR/"last_unet_mask.pth")

        if val_loss<best_val_loss:

            best_val_loss=val_loss
            torch.save(model.state_dict(),CHECKPOINT_DIR/"best_unet_mask.pth")

            print("✅ Best model updated!")

    print("\n🎉 Training completed")

if __name__=="__main__":
    mp.freeze_support()
    main()