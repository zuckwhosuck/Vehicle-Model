import io
import re
import cv2
import torch
import httpx
import asyncio
import numpy as np
from PIL import Image
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production (e.g., ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
async def predict_plates(images: list[UploadFile] = File(...)):
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
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ANPR System Tester</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f3f4f6; padding: 2rem; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
            h1 { color: #1f2937; margin-bottom: 1.5rem; }
            .form-group { margin-bottom: 1.5rem; }
            label { display: block; font-weight: bold; margin-bottom: 0.5rem; color: #4b5563; }
            input[type="text"], input[type="file"] { width: 100%; padding: 0.5rem; border: 1px solid #d1d5db; border-radius: 4px; box-sizing: border-box; }
            button { background-color: #2563eb; color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 4px; font-weight: bold; cursor: pointer; }
            button:hover { background-color: #1d4ed8; }
            #status { margin-top: 1.5rem; padding: 1rem; border-radius: 4px; background-color: #eff6ff; color: #1e40af; display: none; }
            .note { font-size: 0.875rem; color: #6b7280; margin-top: 0.5rem; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>ANPR System Tester</h1>
            <p style="color: #4b5563; margin-bottom: 2rem;">Upload car images to test the FastAPI ANPR backend directly.</p>
            
            <form id="uploadForm">
                <div class="form-group">
                    <label for="batch_id">Batch ID</label>
                    <input type="text" id="batch_id" name="batch_id" value="test_batch_001" required>
                </div>
                
                <div class="form-group">
                    <label for="images">Select Images</label>
                    <input type="file" id="images" name="images" multiple accept="image/*" required>
                </div>

                <button type="submit">Process Images</button>
            </form>

            <div id="status"></div>
            <div id="resultImages" style="margin-top: 2rem; display: flex; flex-direction: column; gap: 1rem;"></div>
        </div>

        <script>
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const form = e.target;
                const statusDiv = document.getElementById('status');
                const resultImages = document.getElementById('resultImages');
                const submitBtn = form.querySelector('button');
                
                submitBtn.disabled = true;
                submitBtn.innerText = "Uploading & Processing...";
                statusDiv.style.display = 'block';
                statusDiv.innerText = "Sending images to the API... (This may take a few seconds)";
                statusDiv.style.backgroundColor = '#eff6ff';
                statusDiv.style.color = '#1e40af';
                resultImages.innerHTML = '';

                const formData = new FormData();
                const files = document.getElementById("images").files;
                for(let i=0; i<files.length; i++) {
                    formData.append("images", files[i]);
                }

                try {
                    const response = await fetch('/api/predict_plates', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    statusDiv.innerText = `Final Plate: ${result.final_plate || 'None'}`;
                    statusDiv.style.backgroundColor = '#ecfdf5';
                    statusDiv.style.color = '#065f46';
                    
                    if (result.images) {
                        result.images.forEach(img => {
                            if (img.blurred_base64) {
                                const imgTag = document.createElement('img');
                                imgTag.src = img.blurred_base64;
                                imgTag.style.maxWidth = '100%';
                                imgTag.style.borderRadius = '8px';
                                imgTag.style.border = '1px solid #d1d5db';
                                
                                const p = document.createElement('p');
                                p.innerText = `${img.filename} - OCR: ${img.prediction || 'N/A'}`;
                                p.style.fontWeight = 'bold';
                                
                                resultImages.appendChild(p);
                                resultImages.appendChild(imgTag);
                            }
                        });
                    }
                } catch (error) {
                    statusDiv.innerText = `Error: ${error.message}`;
                    statusDiv.style.backgroundColor = '#fef2f2';
                    statusDiv.style.color = '#991b1b';
                } finally {
                    submitBtn.disabled = false;
                    submitBtn.innerText = "Process Images";
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
