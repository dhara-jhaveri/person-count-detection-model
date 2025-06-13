import cv2 # OpenCV library
import numpy as np # Numerical Python library
from ultralytics import YOLO # YOLOv8 library

# --- Configuration ---
# 1. Load the pre-trained YOLOv8 model
# 'yolov8n.pt' is the 'nano' version - it's small, fast, and good for real-time webcam use.
# You can try 'yolov8s.pt' for slightly better accuracy but a bit slower.
model = YOLO('yolov8n.pt')

# 2. Define the class ID for 'person' in the COCO dataset
# YOLO models are typically trained on the COCO dataset, where 'person' has an ID of 0.
# You can confirm this by printing model.names, but 0 is standard.
PERSON_CLASS_ID = 0

# 3. Set a confidence threshold
# Only detections with a confidence score above this value will be considered.
# Lowering this might detect more objects but could lead to more false positives.
# Raising it will reduce false positives but might miss some people.QQ
CONF_THRESHOLD = 0.5 # 50% confidence

# --- Main Logic ---

def run_person_counter():
    # Initialize webcam capture

    cap = cv2.VideoCapture(0)

    # Check if the webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam. Make sure it's connected and not in use by another application.")
        return

    print("Webcam opened successfully. Press 'q' to quit.")

    while True:
        # Read a frame from the webcam
        # 'ret' is a boolean indicating if the frame was read successfully.
        # 'frame' is the actual image (a NumPy array).
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # --- Perform Object Detection with YOLOv8 ---
        # The 'model()' call runs the detection on the current frame.
        # 'conf': Only show detections with confidence above CONF_THRESHOLD.
        # 'classes': Only detect the 'person' class (ID 0).
        results = model(frame, conf=CONF_THRESHOLD, classes=[PERSON_CLASS_ID])

        # 'results' will contain information about detected objects.
        # We are interested in the 'boxes' attribute, which holds bounding box info.

        # Initialize a counter for people in the current frame
        people_on_screen = 0

        # Loop through each detected object in the results
        if results and results[0].boxes: # Check if any boxes were detected
            for box in results[0].boxes:
                # Get bounding box coordinates (x1, y1, x2, y2)
                # .xyxy[0] gives the coordinates as [x1, y1, x2, y2]
                # map(int, ...) converts them from float to integer
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Get confidence score and class ID
                confidence = float(box.conf[0]) # Confidence score for this detection
                class_id = int(box.cls[0])     # Class ID (should be 0 for person)

                # Confirm it's a person and draw the bounding box
                if class_id == PERSON_CLASS_ID:
                    people_on_screen += 1
                    # Draw rectangle: cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box (BGR format)

                    # Put label text: cv2.putText(image, text, (x, y), font, scale, color, thickness)
                    label = f'Person: {confidence:.2f}' # e.g., 'Person: 0.92'
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the current count of people on the frame
        count_text = f'People on Screen: {people_on_screen}'
        cv2.putText(frame, count_text, (10, 30), # Position top-left
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2) # Yellow color

        # Display the frame with detections
        cv2.imshow('Live People Counter', frame)

        # Wait for a key press. If 'q' is pressed, break the loop.
        # cv2.waitKey(1) waits for 1 millisecond.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print("Exiting Person Counter.")

# Run the function
if __name__ == '__main__':
    run_person_counter()