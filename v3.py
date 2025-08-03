import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
from sort import Sort
import time
from datetime import datetime
import re
import os

# Load YOLOv8 model (your trained weights)
model = YOLO("best.pt")

# Initialize EasyOCR reader
reader = easyocr.Reader(["en"])

# Initialize SORT tracker
tracker = Sort()

# Improved regex: exactly 2 or 3 letters followed by 3 or 4 digits, no separators
pattern = re.compile(r"^[A-Z]{2,3}[0-9]{3,4}$")

# Create folders
os.makedirs("plates", exist_ok=True)
os.makedirs("ocr", exist_ok=True)

# Video capture
# cap = cv2.VideoCapture("b-3.mp4") # ✅✅✅✅✅
# cap = cv2.VideoCapture("b-2.mp4") #✅✅✅✅✅
# cap = cv2.VideoCapture("my3.mp4") #✅✅✅✅✅
# cap = cv2.VideoCapture("uni-1.mp4") # ✅✅✅✅
# cap = cv2.VideoCapture("uni-2.mp4") # ✅✅✅✅
# cap = cv2.VideoCapture("uni-4.mp4") # ✅✅✅  crop
cap = cv2.VideoCapture("uni-5.mp4") # ✅

fps = cap.get(cv2.CAP_PROP_FPS)

# Trackers
detected_plates = {}  # {track_id: plate_text}
valid_plate_list = set()
saved_raw_ids = set()
saved_ocr_ids = set()

# FPS timing
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = results.boxes.data.cpu().numpy()

    license_plate_class = 0
    detections = detections[detections[:, 5] == license_plate_class]
    detections = detections[:, :5] if len(detections) > 0 else np.empty((0, 5))

    # Tracking
    if detections.shape[0] > 0 and detections.shape[1] >= 5:
        tracks = tracker.update(detections)
    else:
        tracks = []

    # Current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for track in tracks:
        x1, y1, x2, y2, track_id = track
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cropped_plate = frame[y1:y2, x1:x2]

        # Save raw cropped plate (once)
        if track_id not in saved_raw_ids and cropped_plate.size > 0:
            raw_path = f"plates/ID_{int(track_id)}_{timestamp}.jpg"
            cv2.imwrite(raw_path, cropped_plate)
            saved_raw_ids.add(track_id)

        # OCR and regex check
        if track_id not in detected_plates and cropped_plate.size > 0:
            text_results = reader.readtext(cropped_plate)
            # Combine all confident text lines
            plate_lines = []
            for _, text, conf in text_results:
                if conf > 0.5:
                    cleaned = text.strip().upper()
                    plate_lines.append(cleaned)

            # Merge all lines into one string
            combined = "".join(plate_lines)
            normalized = re.sub(r"[^A-Z0-9]", "", combined)

            if pattern.match(normalized):
                detected_plates[track_id] = normalized
                valid_plate_list.add(normalized)

                print(f"[{timestamp}] ✅ ID {int(track_id)} ➤ Plate: {normalized}")

                if track_id not in saved_ocr_ids:
                    ocr_path = f"ocr/PLATE_{normalized}_{timestamp}.jpg"
                    cv2.imwrite(ocr_path, cropped_plate)
                    saved_ocr_ids.add(track_id)
                else:
                    # print(f"[{timestamp}] ❌ Rejected (Invalid Format): {plate_raw}")
                    print(f"[{timestamp}] ❌ Rejected (Invalid Format): ")
                    break  # Only take first high-confidence result

        # Show visual
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID: {int(track_id)}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        if track_id in detected_plates:
            cv2.putText(
                frame,
                detected_plates[track_id],
                (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

    # FPS display
    curr_time = time.time()
    fps_disp = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(
        frame,
        f"FPS: {fps_disp:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    # Timestamp overlay
    cv2.putText(
        frame,
        f"Time: {timestamp}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2,
    )

    # Show video
    cv2.imshow("Number Plate Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Final Output
print("\n✅ All Valid Plates Detected:")
print(sorted(valid_plate_list) if valid_plate_list else "❌ No valid plates detected.")
