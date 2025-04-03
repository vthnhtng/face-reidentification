import numpy as np
from interface.recognizer import Recognizer as RecognizerInterface
from models.arcface import ArcFace as ArcFaceModel

class ArcFace(RecognizerInterface):
    """
    Concrete implementation of the Recognizer interface using ArcFace.
    """

    def __init__(self, weights_path: str):
        self.model = None

        if (weights_path is None) or (weights_path == ""):
            raise ValueError("Invalid ArcFace weights path provided.")
        self.load_model(weights_path)

    def load_model(self, weights_path: str) -> None:
        """
        Load the ArcFace model weights from the specified path.
        
        Args:
            weights_path (str): Path to the ArcFace model weights file.
        """
        print(f"Loading ArcFace model weights from {weights_path}")
        self.model = ArcFaceModel(weights_path)

    def recognize_face(self, image: np.ndarray, keypoint: np.ndarray) -> np.ndarray:
        """
        Recognize a face by extracting an embedding from the image using the provided keypoints.
        
        Args:
            image (np.ndarray): Input image containing the face.
            keypoints (np.ndarray): Keypoints corresponding to facial landmarks.
            
        Returns:
            np.ndarray: The facial embedding (feature vector).
        """
        print("Generate embedding using ArcFace model")
        return self.model(image, keypoint)
