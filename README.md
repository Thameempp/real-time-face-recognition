# Real-Time Face Recognition

Webcam face recognition and dataset collection using local computer-vision models.

## Tools and services

- **Python 3.10+** runs the application and dataset collector.
- **OpenCV** captures webcam frames, displays results, and creates image augmentations.
- **Ultralytics YOLOv8** detects faces using the `yolov8n-face.pt` model.
- **InsightFace `buffalo_s`** creates face embeddings for identity matching.
- **ONNX Runtime** runs InsightFace, using CoreML, CUDA, or CPU providers when available.
- **PyTorch** supports YOLO device selection, preferring CUDA, then Apple MPS, then CPU.
- **NumPy** handles image and embedding arrays.
- **YapaLab GitHub Release** supplies the default YOLO weights, downloaded automatically
  when `models/yolov8n-face.pt` is missing. Use `--no-download` to disable this.

## Setup

Requirements: Python 3.10+, a working webcam, and camera permission.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Workflow

### 1. Collect face images

Run this from the project root:

```bash
python3 -m train --name person_name
```

Omit `--name` to enter the name interactively. Enroll with one person in view
and collect real images across the listed poses, camera distances, expressions,
and lighting conditions. The collector rejects blurry, small, or frame-edge
faces and asks you to try again; it does not enlarge low-resolution crops.

Controls:

```text
1-7  Select pose
s    Save the current face crop
q    Quit
```

Images are saved under `trained_data/person_name/` as originals plus corrected,
dark, bright, and shadow lighting forms. The collector no longer generates
`edges`, `grayscale`, `outline`, or `pattern` forms because the embedding
database does not use them.

### 2. Build embeddings

```bash
python3 main.py --rebuild --build-only
```

This rebuilds the cache and exits without opening the camera. It creates or
updates:

```text
face_embeddings.npz
face_embeddings_manifest.json
```

The manifest records the InsightFace model and version, preprocessing settings,
embedding dimension, and a fingerprint of the enrollment files. Live recognition
refuses a stale or invalid cache; rebuild it explicitly before starting the camera.

### 3. Validate and evaluate

Find exact and near-duplicate original enrollment images. The command reports
pairs for review and does not delete images:

```bash
python3 -m train.validate_dataset
```

Keep three separate, non-overlapping datasets:

```text
trained_data/       # enrollment identities
calibration_data/   # known and unknown identities; threshold selection only
test_data/          # held-out known and unknown identities; final report only
```

Within calibration and test, put each identity in its own subfolder. For unknown
people, use folder names that do not occur under `trained_data/`. The evaluator
compares original-only and selected-augmentation enrollment, top-three mean,
maximum, and normalized prototype strategies. It selects thresholds using only
`calibration_data/`, then reports false-accept and false-reject rates on the
untouched `test_data/`.

```bash
python3 -m recognition.evaluation evaluate-matching \
  --enrollment-dir trained_data \
  --calibration-dir calibration_data \
  --test-dir test_data
```

Benchmark warmed YOLO runs at 416, 512, and 640 on representative images:

```bash
python3 -m recognition.evaluation benchmark-yolo \
  --image-dir benchmark_images \
  --sizes 416 512 640 \
  --device mps
```

The benchmark reports latency percentiles and detected-box counts. Box counts
are not accuracy measurements; use annotated images to compare detection
accuracy at different input sizes.

Compare single-crop InsightFace calls with batched recognition-model inference
while checking embedding agreement:

```bash
python3 -m recognition.evaluation benchmark-embeddings \
  --image-dir trained_data \
  --batch-size 8
```

### 4. Run recognition

```bash
python3 main.py --imgsz 512 --device mps --matching-strategy top3 --threshold 0.45
```

This faster setting suits Apple Silicon systems like the one shown in the screenshot.
Smaller faces may be harder to detect; use `--imgsz 640` when detection of small
or distant faces matters more. On other hardware, omit `--device mps` to let the
application choose the available device. Set `--threshold` to a value selected
from calibration data; `0.45` remains the default until calibration is run.

The app loads the validated embedding cache, opens the webcam, and performs YOLO
detection in a background worker. It rejects low-quality crops, reuses recent
embeddings for stable tracks, and smooths identity decisions across observations.
Press `q` in the camera window to quit.

If `models/yolov8n-face.pt` is missing, it is downloaded automatically. Use `--no-download` to disable this.

## Common options

```bash
python3 main.py --rebuild
python3 main.py --device cpu
python3 main.py --device mps
python3 main.py --device cuda
python3 main.py --conf 0.35
python3 main.py --imgsz 960
python3 main.py --matching-strategy max --threshold 0.50
python3 main.py --rebuild --build-only
python3 main.py --yolo-model models/yolov8n-face.pt
python3 main.py --no-download
```

- `--rebuild`: rebuild the embedding cache and exit.
- `--build-only`: validate the cache and exit; use `--rebuild` to regenerate it.
- `--device`: select `cpu`, `mps`, or `cuda`.
- `--conf`: set the YOLO confidence threshold.
- `--matching-strategy`: choose the per-person top-three average or maximum score.
- `--threshold`: set the minimum accepted identity similarity.
- `--imgsz`: set the YOLO inference image size; lower values are faster, while
  higher values can help detect smaller or more distant faces.
- `--yolo-model`: select a YOLO model file.

## Project structure

```text
.
├── main.py
├── config.py
├── runtime_utils.py
├── face_recognition.py
├── recognition/
│   ├── camera/
│   │   ├── app.py                  # Camera and worker lifecycle
│   │   └── loop.py                 # Live display and result freshness
│   ├── workers/
│   │   ├── recognition.py          # Background recognition worker
│   │   └── buffers.py               # Latest-frame/result queue operations
│   ├── tracking/
│   │   ├── tracker.py              # Face tracks and identity smoothing
│   │   └── geometry.py             # Box overlap and motion geometry
│   ├── evaluation/                 # CLI, data, metrics, galleries, benchmarks
│   ├── rendering/                  # Face labels and performance overlay
│   ├── detector.py                 # YOLO detection and recognition pipeline
│   ├── recognizer.py               # Recognition helper exports
│   ├── face_quality.py             # Face box checks and crop preparation
│   ├── embeddings.py               # InsightFace alignment and embedding batches
│   ├── matching.py                 # Embedding normalization and identity matching
│   ├── drawing.py                  # Compatibility exports for recognition drawing
│   ├── embedding_cache.py          # Cache loading and embedding validation
│   └── model_utils.py              # Device and model setup
├── train/
│   ├── collector/                  # Enrollment app, capture, and display loop
│   ├── display/                    # Collector face drawing and status overlay
│   ├── validation/                 # Duplicate detection helpers
│   ├── dataset.py                  # Enrollment folder and image writing
│   └── validate_dataset.py         # Dataset validation CLI
├── models/yolov8n-face.pt
├── trained_data/
├── face_embeddings.npz
└── face_embeddings_manifest.json
```

## Recognition flow

The recognition flow starts in `main.py`. `recognition/camera/` captures frames
and publishes the latest one through shared memory. `recognition/workers/` runs
YOLO, quality checks, tracking, and identity matching in the background.
`recognition/detector.py` prepares padded crops and rejects blurry, tiny, or
frame-edge faces. Accepted crops are passed at their original resolution to
`InsightFace buffalo_s`, which performs its own landmark alignment and
preprocessing. The aligned crops are sent together through InsightFace's
recognition model when the selected ONNX provider supports batching; otherwise
it falls back to single-crop forwards. Low-resolution crops are not enlarged.
`recognition/tracking/` reuses embeddings for stable tracks and smooths identity changes.
`recognition/rendering/` renders labels, track IDs, and latency and freshness
metrics.

`recognition/embedding_cache.py` loads and validates the embedding cache.
`recognition/model_utils.py` selects a Torch device and checks or downloads the
YOLO model. Cache rebuilding is an explicit
build-only command, separate from live recognition. `runtime_utils.py` selects
available ONNX Runtime providers. The database is built from collected images
by `face_recognition.py`. `recognition.evaluation` benchmarks YOLO sizes and
compares match strategies and enrollment modes with separate calibration and
holdout test sets.

## Configuration

Recognition defaults are defined in `config.py`:

| Setting | Default |
| --- | --- |
| InsightFace model | `buffalo_s` |
| YOLO model | `models/yolov8n-face.pt` |
| YOLO image size / confidence | `640` / `0.25` |
| Maximum detected faces | `20` |
| Identity similarity threshold / strategy | `0.45` / `top3` |
| Requested camera size / frame rate | `1280×720` / `60 FPS` |
| Display-loop FPS cap | `60 FPS` (`None` disables pacing) |
| Recognition crop padding | At least `16 px` or `25%` of face size |
| Minimum face side / blur variance | `48 px` / `18` |
| Frame-edge rejection margin | `4 px` |
| Live InsightFace detection size / threshold | `320×320` / `0.4` |
| Track expiry / embedding refresh | `0.75 s` / `1.0 s` |
| Maximum displayed result age | `0.75 s` |

Dataset collection defaults are defined in `train/config.py`:

| Setting | Default |
| --- | --- |
| Requested camera size / frame rate | `1280×720` / `60 FPS` |
| Detection frame width | `960 px` maximum |
| InsightFace detection size / threshold | `640×640` / `0.35` |
| Collection crop padding | At least `60 px` or `35%` of face size |
| Minimum enrollment face side | `96 px` |
| Save cooldown / maximum result age | `0.5 s` / `1.0 s` |
| Recent frames retained for matching | `12` |

By default, training images are stored in `trained_data/` at the project root.
Set `FACE_TRAINED_DATA_FOLDER` to use another location; both collection and
embedding generation read this variable:

```bash
export FACE_TRAINED_DATA_FOLDER=/path/to/trained_data
```

`CAMERA_FPS` is the requested webcam rate, while `DISPLAY_FPS` caps how often
the recognition window updates. The camera driver and hardware may deliver fewer
frames than requested, so these settings cannot guarantee a particular FPS.
The recognition worker has its own FPS and does not set the display rate.
The overlay also reports YOLO, crop preparation, InsightFace, matching, and
result-queue latency, plus skipped frames and inference errors. YOLO is warmed
before recognition timing begins; InsightFace warms on a usable face when one
is present in the initial frame.
