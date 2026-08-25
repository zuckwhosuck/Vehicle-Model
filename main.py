import io
import re
import cv2
import torch
import httpx
import asyncio
import numpy as np
from PIL import Image
from collections import defaultdict
import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Security, HTTPException, status, Depends
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from torchvision import transforms
from ultralytics import YOLO
from contextlib import asynccontextmanager

from strhub.models.parseq.system import PARSeq


YOLO_MODEL = "ANPR/stage_5_model.pt"
PARSEQ_CKPT = "ANPR/Final_Parseq.ckpt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = None
parseq_model = None
parseq_transform = None

PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load Models at startup
    global yolo_model, parseq_model, parseq_transform
    
    print(f"Using device: {device}")
    print("Loading YOLO...")
    yolo_model = YOLO(YOLO_MODEL)
    print("YOLO Loaded")

    print("Loading PARSeq...")
    parseq_model = PARSeq(
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
    ckpt = torch.load(PARSEQ_CKPT, map_location="cpu", weights_only=False)
    parseq_model.load_state_dict(ckpt["state_dict"])
    parseq_model = parseq_model.to(device)
    parseq_model.eval()
    print("PARSeq Loaded")

    parseq_transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    
    yield
    # Cleanup on shutdown (optional)
    print("Shutting down and cleaning up models...")
    yolo_model = None
    parseq_model = None


app = FastAPI(lifespan=lifespan)

# Restrict origins in production! Loaded strictly from environment variables.
ALLOWED_ORIGINS = os.environ["ALLOWED_ORIGINS"].split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_API_KEY = os.environ["API_KEY"]
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == SECRET_API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )


import base64

def parseq_predict(crop):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = parseq_transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = parseq_model(x)
    probs = logits.softmax(-1)
    preds, confs = parseq_model.tokenizer.decode(probs)
    pred = preds[0]
    conf = float(confs[0].mean().item())
    return pred, conf

def blur_plate(image, box):
    x1, y1, x2, y2 = box
    roi = image[y1:y2, x1:x2]
    blur = cv2.GaussianBlur(roi, (99, 99), 30)
    image[y1:y2, x1:x2] = blur
    return image

@app.post("/api/predict_plates")
async def predict_plates(images: list[UploadFile] = File(...), api_key: str = Depends(verify_api_key)):
    """
    Endpoint for JuniGadi to send bulk images synchronously.
    Returns the final predicted plate and the base64 encoded blurred images.
    """
    votes = defaultdict(float)
    processed_images = []

    for img in images:
        img_bytes = await img.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is None:
            continue

        # YOLO inference
        results = yolo_model.predict(source=image, conf=0.25, verbose=False)
        boxes = results[0].boxes
        
        if len(boxes) == 0:
            # If no plate found, we can just return the original image as base64 or skip
            _, encoded_img = cv2.imencode('.jpg', image)
            b64_img = base64.b64encode(encoded_img).decode('utf-8')
            processed_images.append({
                "filename": img.filename,
                "blurred_base64": f"data:image/jpeg;base64,{b64_img}",
                "plate_found": False
            })
            continue

        # Get best bounding box
        best_idx = int(boxes.conf.argmax().item())
        box = boxes.xyxy[best_idx].cpu().numpy()
        x1, y1, x2, y2 = map(int, box[:4])
        
        # Clamp coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # OCR inference
        pred, conf = parseq_predict(crop)
        
        # Regex check
        if PLATE_REGEX.match(pred):
            votes[pred] += conf
            
        # Blur the image
        blurred_image = blur_plate(image.copy(), (x1, y1, x2, y2))
        _, encoded_img = cv2.imencode('.jpg', blurred_image)
        b64_img = base64.b64encode(encoded_img).decode('utf-8')
        
        processed_images.append({
            "filename": img.filename,
            "blurred_base64": f"data:image/jpeg;base64,{b64_img}",
            "plate_found": True,
            "prediction": pred,
            "confidence": conf
        })
    
    # Vote for final plate
    final_plate = max(votes, key=votes.get) if votes else None
    
    return {
        "final_plate": final_plate,
        "images": processed_images,
        "status": "success"
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": yolo_model is not None and parseq_model is not None}

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)
