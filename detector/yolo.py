import numpy as np
from typing import Tuple
from interface.detector import Detector as DetectorInterface
from ultralytics import YOLO

class Yolo(DetectorInterface):
    """
    Concrete implementation of Detector interface using YOLO.
    """

    def __init__(
        self,
        weights_path: str, 
        confidence_threshold: float
    ):
        self.model = None
        self.confidence_threshold = confidence_threshold

        if (weights_path is None) or (weights_path == ""):
            raise ValueError("Invalid YOLO weights path provided.")
        self.load_model(weights_path)

    def load_model(self, weights_path: str) -> None:
        """
        Load the YOLO model weights from the specified path.
        
        Args:
            weights_path (str): Path to the YOLO model weights file.
        """
        print(f"Loading YOLO model weights from {weights_path}")
        self.model = YOLO(weights_path)

    def detect_face(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect faces in an image and extract bounding boxes and keypoints.
        
        Args:
            image (np.ndarray): Input image.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple where the first element is
            bounding box and the second element is keypoints.
        """
        detections = self.model(image, conf=self.confidence_threshold, verbose=False)
        bboxes = detections[0].boxes.xyxy.cpu().numpy()
        kpss = detections[0].keypoints.xy.cpu().numpy()

        if len(bboxes) == 0 or len(kpss) == 0:
            return np.array([]), np.array([])
        print(kpss[0].shape)
        return bboxes[0], kpss[0]

