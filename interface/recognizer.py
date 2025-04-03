from abc import ABC, abstractmethod
import numpy as np

class Recognizer(ABC):
    """
    Interface for a face recognizer that extracts facial embeddings.
    """

    @abstractmethod
    def load_model(self, weights_path: str) -> None:
        """
        Load the face recognition model weights.
        
        Args:
            weights_path (str): Path to the model weights file.
        """
        pass

    @abstractmethod
    def recognize_face(self, image: np.ndarray, keypoint: np.ndarray) -> np.ndarray:
        """
        Recognize a face by extracting an embedding from the image using the provided keypoints.
        
        Args:
            image (np.ndarray): Input image containing the face.
            keypoints (np.ndarray): Keypoints corresponding to facial landmarks.
            
        Returns:
            np.ndarray: The facial embedding (feature vector).
        """
        pass
