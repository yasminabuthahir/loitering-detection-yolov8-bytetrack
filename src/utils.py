import json
import cv2
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort

def load_config(path="config/config.json"):
    with open(path, "r") as f:
        return json.load(f)

def load_model(model_path):
    if model_path.endswith(".pt"):
        return YOLO(model_path)
    elif model_path.endswith(".onnx"):
        return ort.InferenceSession(model_path)
    else:
        raise ValueError("Unsupported model format")

def is_onnx(model):
    return isinstance(model, ort.InferenceSession)

def draw_polygon(frame, polygon, label=None):
    pts = polygon.reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, (255, 255, 0), 2)
    if label:
        cv2.putText(frame, label, tuple(polygon[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)