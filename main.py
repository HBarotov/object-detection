import cv2
from ultralytics import YOLO

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# Load an image
image_path = "test.jpg"
image = cv2.imread(image_path)

# Perform detection
results = yolo_model(image)

# Draw bounding boxes and display results
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = result.names[int(box.cls[0].item())]
        confidence = box.conf[0]

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{label}: {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

cv2.imshow("Detection", image)
cv2.waitKey(0)
