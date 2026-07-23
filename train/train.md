# Train Files

Minimal guide for the face dataset collection section.

## Run

```bash
python3 train.py --name person_name
```

From the project root, this also works:

```bash
python3 -m train --name person_name
```

## Files

- `__init__.py` - Marks `train/` as a Python package.
- `__main__.py` - Allows running the package with `python3 -m train`.
- `main.py` - Main entry point for training image collection. Parses the person name and starts the collector.
- `train.py` - Compatibility wrapper for the old command. It simply calls `train/main.py`.
- `config.py` - Training settings such as camera size, save cooldown, folder names, pose keys, and output path.
- `utils.py` - Small helpers for cleaning person names and keeping only the latest queue item.
- `augmentations.py` - Creates saved image versions: original, corrected, dark, bright, shadow, grayscale, edges, outline, and pattern.
- `face_processing.py` - Resizes frames for detection, converts detected faces to result data, crops faces, and upscales small crops.
- `detector.py` - Background InsightFace detector worker that finds faces in the newest camera frame.
- `dataset.py` - Creates person folders, finds the next image number, and saves all augmented face images.
- `display.py` - Draws detected boxes, keypoints, status text, FPS, pose, and controls on the camera window.
- `collector.py` - Main webcam collection loop. Handles pose keys, save key, recent frames, detection results, and cleanup.

## Output

Collected images are saved under:

```text
../trained_data/person_name/
```

After collecting images, rebuild recognition embeddings:

```bash
python3 ../main.py --rebuild --build-only
```
