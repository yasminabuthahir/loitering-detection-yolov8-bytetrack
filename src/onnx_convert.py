from ultralytics import YOLO
import os

MODEL_PATH = "models/yolov8n.pt"

def convert():
    model = YOLO(MODEL_PATH)

    # Export to ONNX
    model.export(format="onnx", opset=12)

    print("ONNX model exported successfully")

if __name__ == "__main__":
    convert()