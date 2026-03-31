import cv2
import time
import numpy as np
from datetime import datetime
from src.utils import load_config, load_model, draw_polygon
from src.db import insert_event
import sys
import os
from types import SimpleNamespace

# Add ByteTrack root directory to sys.path
bytetrack_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tracking", "ByteTrack"
)
sys.path.insert(0, bytetrack_path)

# Import ByteTrack tracker
from yolox.tracker.byte_tracker import BYTETracker

class LoiteringDetector:
    def __init__(self):
        self.config = load_config()
        self.model = load_model(self.config["model_path"])  # ONNXSession
        self.cap = cv2.VideoCapture(self.config["camera"]["source"])
        self.zone = np.array(self.config["zone"]["polygon"], dtype=np.int32)
        self.threshold = self.config["loiter_threshold_sec"]

        # Dummy args for BYTETracker
        args = SimpleNamespace(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8,
            min_box_area=10
        )
        self.tracker = BYTETracker(args, frame_rate=int(self.cap.get(cv2.CAP_PROP_FPS)))

        self.entry_time = {}      # track_id -> timestamp
        self.loiter_saved = {}    # track_id -> bool

        # Ensure snapshot dir exists
        self.snapshot_dir = self.config["output"]["snapshot_dir"]
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def is_inside(self, point):
        """Check if point is inside ROI polygon"""
        return cv2.pointPolygonTest(self.zone, point, False) >= 0

    def run(self):
        import onnxruntime as ort
        from collections import defaultdict

        # Track entry times and loitering status using centroids
        self.entry_time = defaultdict(float)
        self.loiter_saved = defaultdict(bool)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # --- Preprocess frame for YOLOX ONNX (640x640 model) ---
            img = cv2.resize(frame, (640, 640))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            img = np.expand_dims(img, axis=0)

            # Run ONNX inference
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: img})  # shape (1, 84, 8400)

            # --- Parse detections: only human class 0 ---
            preds = outputs[0][0].T  # (84, 8400) -> (num_boxes, 84)
            detections = []
            for pred in preds:
                cls_id = int(pred[4:].argmax())  # class with highest score
                conf = pred[4 + cls_id]
                if cls_id != 0 or conf < 0.5:
                    continue

                # Convert YOLOX box format (x_center, y_center, w, h) -> (x1, y1, x2, y2)
                x_center, y_center, w, h = pred[0], pred[1], pred[2], pred[3]
                x1 = int(x_center - w / 2)
                y1 = int(y_center - h / 2)
                x2 = int(x_center + w / 2)
                y2 = int(y_center + h / 2)
                detections.append([x1, y1, x2, y2, float(conf)])

            # --- Centroid-based loitering detection ---
            for det in detections:
                x1, y1, x2, y2, conf = det
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                centroid = (cx, cy)

                if self.is_inside(centroid):
                    if self.entry_time[centroid] == 0:
                        self.entry_time[centroid] = time.time()
                        self.loiter_saved[centroid] = False
                    else:
                        duration = time.time() - self.entry_time[centroid]
                        if duration > self.threshold and not self.loiter_saved[centroid]:
                            # Save snapshot
                            filename = f"{self.snapshot_dir}/loitering_{cx}_{cy}_{int(time.time())}.jpg"
                            cv2.imwrite(filename, frame)

                            # Insert event into DB
                            insert_event({
                                "track_id": f"{cx}_{cy}",
                                "duration": duration,
                                "timestamp": datetime.now().isoformat(),
                                "snapshot_path": filename
                            })
                            print(f"Loitering detected. Saved: {filename}")
                            self.loiter_saved[centroid] = True
                else:
                    # Reset if outside ROI
                    self.entry_time.pop(centroid, None)
                    self.loiter_saved.pop(centroid, None)

                # Draw bounding box + centroid
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # Draw ROI polygon
            draw_polygon(frame, self.zone, "ROI")

            # Show frame
            cv2.imshow("Loitering Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break

        self.cap.release()
        cv2.destroyAllWindows()