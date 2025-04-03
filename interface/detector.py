from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

class Detector(ABC):
    """
    Interface for a face detector.
    """

    @abstractmethod
    def load_model(self, weights_path: str) -> None:
        """
        Load the detection model weights.
        
        Args:
            weights_path (str): Path to the model weights file.
        """
        pass

    @abstractmethod
    def detect_face(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect faces in an image and extract bounding boxes and keypoints.
        
        This method is expected to use a detector (e.g., YOLO) to obtain the face
        bounding boxes and keypoints.
        
        Args:
            image (np.ndarray): Input image.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple where the first element is an array 
            of bounding boxes (with each box defined as (x1, y1, x2, y2)) and the second 
            element is an array of keypoints.
        """
        pass
