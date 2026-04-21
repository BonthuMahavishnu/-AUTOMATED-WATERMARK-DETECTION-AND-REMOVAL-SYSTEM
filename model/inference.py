import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from pathlib import Path
from .unet import UNet

# -----------------------------
# DEVICE
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT = BASE_DIR / "checkpoints" / "best_unet_mask.pth"

# -----------------------------
# LOAD MODEL ONCE
# -----------------------------
def load_model():
    model = UNet().to(DEVICE)
    state = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

MODEL = load_model()

# -----------------------------
# REMOVE WATERMARK
# -----------------------------
def remove_watermark(input_path, output_path, size=(256,256)):

    img_bgr = cv2.imread(str(input_path))
    h, w = img_bgr.shape[:2]

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize(size)

    tensor = TF.to_tensor(pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = torch.sigmoid(MODEL(tensor))[0,0].cpu().numpy()

    # -----------------------------
    # MASK GENERATION
    # -----------------------------
    mask = (prob > 0.35).astype(np.uint8)*255
    mask = cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)

    # -----------------------------
    # MASK REFINEMENT
    # -----------------------------
    kernel = np.ones((7,7),np.uint8)

    mask = cv2.dilate(mask,kernel,iterations=2)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    mask = cv2.GaussianBlur(mask,(5,5),0)

    # -----------------------------
    # ADVANCED INPAINT
    # -----------------------------
    result = cv2.inpaint(
        img_bgr,
        mask,
        7,
        cv2.INPAINT_NS
    )

    cv2.imwrite(str(output_path),result)

    return output_path