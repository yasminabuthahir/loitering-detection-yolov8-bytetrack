import cv2
import time
import numpy as np
np.float = float
np.int = int
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
            min_box_area=10,
            mot20=False
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

        def decode_outputs(outputs, img_size):
            grids = []
            expanded_strides = []

            strides = [8, 16, 32]
            hsizes = [img_size[0] // s for s in strides]
            wsizes = [img_size[1] // s for s in strides]

            for hsize, wsize, stride in zip(hsizes, wsizes, strides):
                xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
                grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
                grids.append(grid)
                expanded_strides.append(np.full((*grid.shape[:2], 1), stride))

            grids = np.concatenate(grids, 1)
            expanded_strides = np.concatenate(expanded_strides, 1)

            outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
            outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

            return outputs

        def nms(boxes, scores, iou_threshold=0.5):
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(),
                scores.tolist(),
                score_threshold=0.5,
                nms_threshold=iou_threshold
            )
            if len(indices) == 0:
                return []
            return indices.flatten()

        self.entry_time = defaultdict(float)
        self.loiter_saved = defaultdict(bool)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            # --- Preprocess ---
            img = cv2.resize(frame, (640, 640))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)

            # --- Inference ---
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: img})

            # --- Decode YOLOX ---
            preds = outputs[0][0].T

            boxes = []
            scores = []

            for pred in preds:
                cls_id = int(pred[4:].argmax())
                score = pred[4 + cls_id]

                if cls_id != 0 or score < 0.5:
                    continue

                x_center, y_center, bw, bh = pred[0], pred[1], pred[2], pred[3]

                scale_x = w / 640
                scale_y = h / 640

                x1 = (x_center - bw / 2) * scale_x
                y1 = (y_center - bh / 2) * scale_y
                bw = bw * scale_x
                bh = bh * scale_y

                boxes.append([int(x1), int(y1), int(bw), int(bh)])
                scores.append(float(score))

            # --- NMS ---
            if len(boxes) > 0:
                indices = nms(np.array(boxes), np.array(scores))
            else:
                indices = []

            detections = []
            for i in indices:
                x, y, bw, bh = boxes[i]
                score = scores[i]

                x2 = x + bw
                y2 = y + bh

                detections.append([x, y, x2, y2, score])

            # --- ByteTrack ---
            if len(detections) > 0:
                dets = np.array(detections)
            else:
                dets = np.empty((0, 5))

            online_targets = self.tracker.update(dets, (h, w), (h, w))

            for t in online_targets:
                tlwh = t.tlwh
                track_id = t.track_id

                x1, y1, bw, bh = map(int, tlwh)
                x2 = x1 + bw
                y2 = y1 + bh

                cx, cy = x1 + bw // 2, y1 + bh // 2

                # --- Loitering logic ---
                if self.is_inside((cx, cy)):
                    if self.entry_time[track_id] == 0:
                        self.entry_time[track_id] = time.time()
                        self.loiter_saved[track_id] = False
                    else:
                        duration = time.time() - self.entry_time[track_id]

                        if duration > self.threshold and not self.loiter_saved[track_id]:
                            filename = f"{self.snapshot_dir}/loitering_{track_id}_{int(time.time())}.jpg"
                            cv2.imwrite(filename, frame)

                            insert_event(
                                track_id,
                                duration,
                                datetime.now().isoformat(),
                                filename
                            )

                            print(f"Loitering detected (ID {track_id}). Saved: {filename}")
                            self.loiter_saved[track_id] = True
                else:
                    self.entry_time.pop(track_id, None)
                    self.loiter_saved.pop(track_id, None)

                # --- Draw ---
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            draw_polygon(frame, self.zone, "ROI")

            cv2.imshow("Loitering Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()