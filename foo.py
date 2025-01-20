from modules.detectors.yolo import YoloDetector
from utils.config_loader import ConfigLoader
import cv2

config_loader = ConfigLoader()
detector = YoloDetector(config_loader.get_yolo_weight())

image_path = "6th-street.jpg"  # Provide the correct image path

# Load the image
image = cv2.imread(image_path)

# Perform detection
bboxes, keypoints = detector.detect(image)

# Print the results
print("Bounding Boxes:", bboxes)
print("Keypoints:", keypoints)

# Optional: Visualize the results using OpenCV
for bbox in bboxes:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

# Show the image with bounding boxes
cv2.imshow("Detected Faces", image)
cv2.waitKey(0)
