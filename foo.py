import cv2
from ultralytics import YOLO

# Initialize YOLOv8 model
model = YOLO('yolov8n-face.pt')  # or use your custom trained model

# Read single image
image_path = 'faces/Tung.png'  # Replace with your image path
frame = cv2.imread(image_path)



# Run YOLOv8 inference on the image
results = model(frame)

# Visualize the results on the frame
print(results[0].keypoints.cpu().numpy().xy[0].shape)

cv2.destroyAllWindows()