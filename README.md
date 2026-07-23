# Face Recognition Project

Webcam-based face dataset collection and live face recognition using:

- InsightFace `buffalo_s` for face embeddings
- YOLOv8 face detection for live recognition
- OpenCV for camera capture, display, and image augmentation
- Multiprocessing with shared memory for smoother camera performance

The project has two main workflows:

1. Collect face crops for a person into `trained_data/`
2. Build embeddings and run live recognition from the webcam

## Project Structure

```text
.
├── main.py                         # Live face recognition entry point
├── face_reco_yolo.py               # Compatibility wrapper for main.py
├── face_recognition.py             # Builds and loads embedding cache
├── camera.py                       # Webcam loop for live recognition
├── detector.py                     # YOLO detection + recognition pipeline
├── recognizer.py                   # Face crop embedding and identity matching
├── model_utils.py                  # Model loading, device selection, downloads
├── drawing.py                      # Live recognition overlay drawing
├── config.py                       # Runtime paths and recognition settings
├── train/
│   ├── main.py                     # Dataset collector entry point
│   ├── train.py                    # Compatibility wrapper
│   ├── collector.py                # Webcam collection loop
│   ├── detector.py                 # InsightFace detection worker
│   ├── dataset.py                  # Folder creation and image saving
│   ├── augmentations.py            # Saved image variants
│   ├── display.py                  # Collector overlay and controls
│   └── config.py                   # Training settings
├── models/
│   └── yolov8n-face.pt             # YOLO face model
├── trained_data/                   # Person folders and collected images
├── face_embeddings.npz             # Cached embeddings
└── face_embeddings_manifest.json   # Tracks images used for the cache
```

## Requirements

- Python 3.10 or newer recommended
- A working webcam
- macOS, Linux, or Windows with OpenCV camera support

Install the Python packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install opencv-python numpy insightface onnxruntime ultralytics torch
```

On Apple Silicon, the code will prefer CoreML for InsightFace when available and MPS for Torch/YOLO when available. On NVIDIA systems, it will use CUDA when the installed Torch and ONNX Runtime builds support it. Otherwise, it falls back to CPU.

## Collect Training Images

Run the collector from the project root:

```bash
python3 -m train --name person_name
```

You can also run the compatibility wrapper:

```bash
python3 train/train.py --name person_name
```

If `--name` is omitted, the script asks for the person's name interactively.

### Collector Controls

```text
1 = Front
2 = Slight left
3 = Strong left
4 = Slight right
5 = Strong right
6 = Looking up
7 = Looking down
S = Save current face crop
Q = Quit
```

Keep only one person visible while saving. Each save writes the original crop plus augmented variants under:

```text
trained_data/person_name/
```

The generated folders include:

```text
originals/
corrected/
dark/
bright/
shadow/
grayscale/
edges/
outline/
pattern/
```

Only these folders are used when building embeddings:

```text
originals/
corrected/
dark/
bright/
shadow/
```

## Build or Rebuild Embeddings

After collecting images, rebuild the embedding cache:

```bash
python3 main.py --rebuild --build-only
```

This creates or updates:

```text
face_embeddings.npz
face_embeddings_manifest.json
```

The manifest lets the app detect when training images have changed and rebuild the cache when needed.

## Run Live Recognition

Start the webcam recognition app:

```bash
python3 main.py
```

The app loads cached embeddings, starts the camera, detects faces with YOLO, embeds each detected face with InsightFace, and labels matches in the OpenCV window.

Press `q` in the camera window to quit.

## Useful Runtime Options

```bash
python3 main.py --rebuild
python3 main.py --build-only
python3 main.py --device cpu
python3 main.py --device mps
python3 main.py --device cuda
python3 main.py --conf 0.35
python3 main.py --imgsz 640
python3 main.py --yolo-model models/yolov8n-face.pt
python3 main.py --no-download
```

Options:

- `--rebuild` rebuilds the InsightFace embedding cache before running.
- `--build-only` rebuilds or verifies embeddings and exits without opening the camera.
- `--device` overrides automatic YOLO device selection.
- `--conf` changes the YOLO face confidence threshold.
- `--imgsz` changes the YOLO inference image size.
- `--yolo-model` uses a custom YOLO face model path.
- `--no-download` prevents automatic download of the default YOLO model.

If `models/yolov8n-face.pt` is missing, `main.py` downloads the default model automatically unless `--no-download` is set.

## Configuration

Main recognition settings live in `config.py`, including:

- Camera size and FPS
- YOLO image size and confidence
- InsightFace model name
- Similarity threshold
- Embedding cache paths

Training settings live in `train/config.py`, including:

- Dataset output path
- Pose key mappings
- Save cooldown
- Face crop padding
- Augmentation folder names

To store training images somewhere else, set:

```bash
export FACE_TRAINED_DATA_FOLDER=/path/to/trained_data
```

Both the training and embedding-building code read this environment variable.

## Troubleshooting

If the camera does not open, check that no other app is using it and that your OS has granted camera permission to the terminal.

If no embeddings are found, collect images first and then run:

```bash
python3 main.py --rebuild --build-only
```

If recognition labels are too strict or too loose, tune `SIMILARITY_THRESHOLD` in `config.py`.

If YOLO runs slowly, try a smaller image size:

```bash
python3 main.py --imgsz 640
```

## Notes

This repository currently does not include a pinned dependency file such as `requirements.txt`, so package versions are resolved by `pip` at install time. For reproducible setup across machines, add a `requirements.txt` after confirming the versions that work best on your hardware.
