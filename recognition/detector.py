from ultralytics import YOLO

from config import MAX_FACES
from recognizer import crop_face, embed_crop, recognize_embedding


class YoloFaceRecognizer:
    def __init__(
        self,
        model_path,
        embedder_app,
        known_embeddings,
        known_names,
        image_size,
        confidence,
        device,
    ):
        self.model = YOLO(model_path)
        self.embedder_app = embedder_app
        self.known_embeddings = known_embeddings
        self.known_names = known_names
        self.image_size = image_size
        self.confidence = confidence
        self.device = device

    def detect_and_recognize(self, frame):
        yolo_results = self.model.predict(
            frame,
            imgsz=self.image_size,
            conf=self.confidence,
            device=self.device,
            max_det=MAX_FACES,
            verbose=False,
        )

        detections = []

        if not yolo_results:
            return detections

        boxes = yolo_results[0].boxes

        if boxes is None:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()

        for box, confidence in zip(xyxy, confidences):
            crop = crop_face(frame, box)
            embedding = embed_crop(self.embedder_app, crop)

            if embedding is None:
                name = "Unknown"
                score = 0.0
            else:
                name, score = recognize_embedding(
                    embedding,
                    self.known_embeddings,
                    self.known_names,
                )

            detections.append(
                {
                    "bbox": box.tolist(),
                    "name": name,
                    "score": float(score),
                    "det_conf": float(confidence),
                }
            )

        return detections
