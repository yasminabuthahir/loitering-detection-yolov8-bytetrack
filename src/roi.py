import cv2
import os
import numpy as np

VIDEO_PATH = "data/loitering.mp4"
OUTPUT_PATH = "output/roi_frame.jpg"

# 👉 PUT YOUR COORDINATES HERE
ROI_POINTS = [
    [120, 60],
    [300, 120],
    [300, 230],
    [5, 100]
]

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error opening video")
        return

    frame = None

    # Get 4th frame
    for i in range(4):
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            return

    cap.release()

    # Convert ROI to numpy
    polygon = np.array(ROI_POINTS, dtype=np.int32)

    # Draw polygon
    cv2.polylines(frame, [polygon], isClosed=True, color=(255, 255, 0), thickness=2)

    # Draw points
    for i, (x, y) in enumerate(ROI_POINTS):
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"P{i+1}", (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Ensure output folder exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Save image
    cv2.imwrite(OUTPUT_PATH, frame)

    print(f"Saved ROI frame at: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()