import cv2
import torch
import numpy as np
from ultralytics import YOLO
from playerTracker import PlayerTrackerCNN  # Ensure this file exists and is correct

# Load the YOLOv8 model
model = YOLO("best.pt")  # Ensure this is the correct path

def run_yolo(frame):
    """
    Runs YOLOv8 on the input frame and returns all detections.
    Each detection is a tuple: (class_id, [x1, y1, x2, y2])
    """
    results = model(frame, verbose=False)[0]
    detections = []

    for box in results.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append((cls, [x1, y1, x2, y2]))

    return detections

def main():
    # Open the video
    cap = cv2.VideoCapture("15sec_input_720p.mp4")
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Initialize the tracker
    tracker = PlayerTrackerCNN()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: Detect all objects
        detections = run_yolo(frame)

        # Step 2: Filter out non-human classes for tracking (skip ball)
        filtered = [(cls, bbox) for cls, bbox in detections if cls in [1, 2, 3]]  # goalkeeper, player, referee
        matched_players = tracker.match_players(frame, filtered)

        # Step 3: Draw tracked humans with IDs
        for pid, (cls, (x1, y1, x2, y2)) in matched_players:
            label = model.names[cls]
            # Choose color based on class
            color = (0, 255, 0) if label == "player" else \
                    (255, 0, 0) if label == "goalkeeper" else \
                    (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {pid}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Step 4: Optionally draw ball separately
        for cls, (x1, y1, x2, y2) in detections:
            if cls == 0:  # ball
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 10, (0, 255, 255), -1)
                cv2.putText(frame, "ball", (cx - 10, cy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Step 5: Show frame
        cv2.imshow("Player Re-Identification", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
