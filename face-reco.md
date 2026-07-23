# Face Recognition Files

Minimal guide for the live face recognition section.

## Run

```bash
python3 main.py
```

Rebuild embeddings only:

```bash
python3 main.py --rebuild --build-only
```

## Files

- `main.py` - Main entry point for live face recognition. Parses arguments, loads embeddings, checks the YOLO model, and starts the camera.
- `config.py` - Shared settings such as paths, camera size, YOLO confidence, model names, and recognition thresholds.
- `camera.py` - Opens the webcam, shares frames with the worker process, receives recognition results, and displays the final window.
- `detector.py` - Runs YOLO face detection and sends each face crop to the recognizer.
- `recognizer.py` - Crops faces, prepares small face images, creates embeddings, and matches them against known embeddings.
- `workers.py` - Background multiprocessing worker that reads the newest frame and performs detection plus recognition.
- `drawing.py` - Draws face boxes, labels, FPS, device info, and status text on the video frame.
- `model_utils.py` - Loads InsightFace, chooses CPU/CoreML/CUDA providers, chooses the Torch device, loads embeddings, and downloads/checks the YOLO model.
- `face_recognition.py` - Builds and loads the face embedding database from `trained_data/`.
- `face_reco_yolo.py` - Compatibility wrapper for the old command. It simply calls `main.py`.
- `face_embeddings.npz` - Cached face embeddings and names used during recognition.
- `face_embeddings_manifest.json` - Tracks training images used to build the embedding cache.
- `models/yolov8n-face.pt` - YOLO face detection model.
- `trained_data/` - Person image folders collected by the training script.
