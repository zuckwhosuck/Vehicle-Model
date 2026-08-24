# Junigadi ANPR System

This project contains the Automatic Number Plate Recognition (ANPR) system for Junigadi. It leverages a two-stage deep learning pipeline to detect and recognize Indian license plates from vehicle images or video frames.

## 🚀 How It Works (The Pipeline)

The system operates in two separate stages using two different AI models:

### Stage 1: License Plate Detection (YOLOv8)
- **Input**: An image of a car/road.
- **Model**: YOLOv8 (`stage_5_model.pt`).
- **Process**: The model scans the image and outputs the exact coordinates (bounding box) where the license plate is located.
- **Setup**: It picks the highest confidence box and crops that region from the image.

### Stage 2: Character Recognition (PARSeq OCR)
- **Input**: The cropped plate image from Stage 1.
- **Model**: PARSeq (`best_parseq.ckpt`).
- **Process**: The crop is resized to 32x128 pixels, normalized, and read by the PARSeq model.
- **Output**: The actual license plate text (e.g., "GJ01AB1234") and a confidence score.

### Multi-Frame Voting
When processing video frames or multiple images of the same car, the script (`number_plate_reader.py`) counts how many times different texts are predicted across frames and weights them by confidence. The text with the highest accumulated score is selected as the final plate.

## 📁 Repository Structure
- `number_plate_reader.py`: The main Python script that reads images from a directory, runs the YOLO + PARSeq pipeline, blurs valid plates, and performs voting to determine the most probable license plate.
- `requirements.txt`: The Python dependencies required to run the pipeline.
- `ANPR_DEPLOYMENT.txt`: Detailed deployment and architecture guide.

## 🛠 Hardware & Software Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support is highly recommended for production. It reduces processing time to under 50-100ms per image.
- **CPU**: A standard CPU works but is much slower (200-500ms per image).

### Software
It is highly recommended to use a Python virtual environment to manage dependencies:

```bash
# 1. Create a virtual environment
python3 -m venv venv

# 2. Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 3. Install the required packages
pip install -r requirements.txt
```

**Key Dependencies**:
- Python 3.10+
- `torch` & `torchvision`
- `ultralytics` (YOLO)
- `parseq` (strhub / PARSeq)
- `opencv-python`
- `Pillow`, `numpy`, `tqdm`
- `fastapi`, `uvicorn` (for backend deployment)

## 🔒 Privacy & Plate Blurring

To protect privacy, the system can blur the license plate in the image. To prevent false blurring (e.g., blurring car vents or signs mistakenly detected as plates):
- The bounding box is ONLY blurred if the PARSeq OCR successfully reads readable text in that box.
- A regular expression validation check (`PLATE_REGEX`) ensures the text matches standard Indian plate structures (e.g., `^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$`).
- If the text fails to match the expected pattern, the box is NOT blurred.

## 🌐 Production Architecture & Implemented API

The system is deployed using a **FastAPI Backend API Service** which runs asynchronously in the background.

1. **Python API Backend (`main.py`)**: 
   - A lightweight FastAPI server that loads the YOLO and PARSeq models once on startup, keeping them active in memory.
   - It exposes `POST /api/predict_plates` which accepts bulk image uploads and a `callback_url`.
   - The server immediately acknowledges the request and processes the images in a background task. Once finished, it sends a webhook POST request to the provided callback URL with the final JSON results.
   - It also provides a root `GET /` endpoint that serves an HTML testing GUI.

2. **Starting the Server**:
   To start the backend server (we recommend using port `8080` to avoid conflicts with background agents):
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8080
   ```
   Once started, you can access the testing interface at [http://localhost:8080/](http://localhost:8080/).

3. **Web Frontend (JuniGadi)**: 
   The Next.js website sends bulk images to the Python endpoint via `fetch()` and listens for the background JSON webhook on its own API route.

## 🧠 Self-Learning & Continuous Improvement

To make the model self-learning, implement an automated feedback loop:
1. **Collect Data**: Log hard examples (low confidence) and user-corrected predictions.
2. **Admin Review**: Build a dashboard for a human operator to validate or correct the collected images.
3. **Periodic Retraining**: Run YOLO and PARSeq training scripts using this newly audited data to update the model weights (`stage_5_model.pt` and `best_parseq.ckpt`), then deploy the new models to the backend API server.
# Vehicle-Model
