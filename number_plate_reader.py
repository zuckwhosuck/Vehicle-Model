import re
from pathlib import Path
from collections import defaultdict

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from strhub.models.parseq.system import PARSeq


# =====================================================
# PATHS
# =====================================================

YOLO_MODEL = r"ANPR/stage_5_model.pt"

PARSEQ_CKPT = r"ANPR/Final_Parseq.ckpt"

ROOT_DIR = Path(
    r"C:\Users\VANSH\Desktop\Junigadi\test"
)

CAR_FOLDERS = ["car2"]


# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# =====================================================
# LOAD YOLO
# =====================================================

print("Loading YOLO...")

yolo = YOLO(YOLO_MODEL)

print("YOLO Loaded")


# =====================================================
# LOAD PARSEQ
# =====================================================

print("Loading PARSeq...")

model = PARSeq(
    charset_train="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    charset_test="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",

    max_label_length=25,

    batch_size=16,
    lr=0.0007,
    warmup_pct=0.075,
    weight_decay=0.0,

    img_size=(32, 128),
    patch_size=(4, 8),

    embed_dim=384,
    enc_num_heads=6,
    enc_mlp_ratio=4,
    enc_depth=12,

    dec_num_heads=12,
    dec_mlp_ratio=4,
    dec_depth=1,

    perm_num=6,
    perm_forward=True,
    perm_mirrored=True,

    decode_ar=True,
    refine_iters=1,

    dropout=0.1
)

ckpt = torch.load(
    PARSEQ_CKPT,
    map_location="cpu"
)

model.load_state_dict(
    ckpt["state_dict"]
)

model = model.to(device)
model.eval()

print("PARSeq Loaded")


# =====================================================
# OCR TRANSFORM
# =====================================================

transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])


# =====================================================
# REGEX
# =====================================================

PLATE_REGEX = re.compile(
    r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$"
)


# =====================================================
# OCR FUNCTION
# =====================================================

def parseq_predict(crop):

    rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    pil = Image.fromarray(rgb)

    x = transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)

    probs = logits.softmax(-1)

    preds, confs = model.tokenizer.decode(probs)

    pred = preds[0]

    conf = float(
        confs[0].mean().item()
    )

    return pred, conf


# =====================================================
# PROCESS EACH CAR
# =====================================================

for car_name in CAR_FOLDERS:

    print("\n" + "=" * 60)
    print("PROCESSING:", car_name)
    print("=" * 60)

    car_dir = ROOT_DIR / car_name

    if not car_dir.exists():
        print("Missing:", car_dir)
        continue

    blur_dir = ROOT_DIR / f"{car_name}_blurred"

    blur_dir.mkdir(
        exist_ok=True
    )

    votes = defaultdict(float)

    image_count = 0

    images = []

    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
        images.extend(car_dir.glob(ext))

    print("Images found:", len(images))

    for img_path in images:

        image_count += 1

        image = cv2.imread(str(img_path))

        if image is None:
            continue

        results = yolo.predict(
            source=image,
            conf=0.25,
            verbose=False
        )

        boxes = results[0].boxes

        if len(boxes) == 0:

            cv2.imwrite(
                str(blur_dir / img_path.name),
                image
            )

            continue

        # ==================================
        # BEST BOX ONLY
        # ==================================

        best_idx = int(
            boxes.conf.argmax().item()
        )

        box = boxes.xyxy[
            best_idx
        ].cpu().numpy()

        x1, y1, x2, y2 = map(
            int,
            box[:4]
        )

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        crop = image[
            y1:y2,
            x1:x2
        ]

        if crop.size == 0:

            cv2.imwrite(
                str(blur_dir / img_path.name),
                image
            )

            continue

        # ==================================
        # OCR
        # ==================================

        pred, conf = parseq_predict(crop)

        print(
            f"{img_path.name:40s} "
            f"-> {pred:15s} "
            f"({conf:.3f})"
        )

        # ==================================
        # REGEX FILTER
        # ==================================

        if PLATE_REGEX.match(pred):
            votes[pred] += conf

        # ==================================
        # BLUR
        # ==================================

        roi = image[y1:y2, x1:x2]

        blur = cv2.GaussianBlur(
            roi,
            (99, 99),
            30
        )

        image[y1:y2, x1:x2] = blur

        cv2.imwrite(
            str(blur_dir / img_path.name),
            image
        )

    # ======================================
    # FINAL VOTE
    # ======================================

    if len(votes) == 0:

        print("\nFINAL PLATE : NOT FOUND")

    else:

        final_plate = max(
            votes,
            key=votes.get
        )

        print("\n")
        print("=" * 60)
        print("FINAL PLATE :", final_plate)
        print("=" * 60)

        print("\nVote Scores:")

        for k, v in sorted(
            votes.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(
                f"{k:15s}  {v:.3f}"
            )

    print(
        "\nBlurred images saved to:",
        blur_dir
    )

print("\nDONE")