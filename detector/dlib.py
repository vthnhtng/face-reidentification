import numpy as np
import dlib
from typing import Tuple
from interface.detector import Detector as DetectorInterface


class Dlib(DetectorInterface):
    """
    Concrete implementation of Detector interface using dlib.
    """

    def __init__(
        self,
        weights_path: str,
        confidence_threshold: float = 0.0  # Not used in dlib but kept for compatibility
    ):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = None
        self.confidence_threshold = confidence_threshold  # Not applicable but stored

        if not weights_path:
            raise ValueError("Invalid dlib shape predictor path provided.")
        self.load_model(weights_path)

    def load_model(self, weights_path: str) -> None:
        """
        Load the dlib shape predictor model from the specified path.
        
        Args:
            weights_path (str): Path to the shape predictor .dat file.
        """
        print(f"Loading dlib shape predictor from {weights_path}")
        self.predictor = dlib.shape_predictor(weights_path)

    def detect_face(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect a face in an image and extract the bounding box and landmarks.
        
        Args:
            image (np.ndarray): Input image in BGR format.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Bounding box (x1, y1, x2, y2) and keypoints (68, 2).
        """
        # Convert BGR to RGB
        img_rgb = image[:, :, ::-1]

        faces = self.detector(img_rgb, 1)
        if not faces:
            return np.array([]), np.array([])

        # Use the first detected face
        face = faces[0]
        shape = self.predictor(img_rgb, face)

        # Bounding box
        bbox = np.array([face.left(), face.top(), face.right(), face.bottom()], dtype=np.float32)

        # Keypoints
        keypoints = np.zeros((68, 2), dtype=np.float32)
        for i in range(68):
            part = shape.part(i)
            keypoints[i] = (part.x, part.y)

        print(keypoints.shape)
        return bbox, self.extract_five_landmarks(keypoints)

    def extract_five_landmarks(self, shape_68: np.ndarray) -> np.ndarray:
        """
        Extract 5 facial landmarks from 68 landmarks.
        
        Parameters:
            shape_68 (np.ndarray): An array of shape (68, 2) containing dlib facial landmarks.
            
        Returns:
            shape_5 (np.ndarray): An array of shape (5, 2) containing 5 key landmarks:
                [left_eye_center, right_eye_center, nose_tip, left_mouth_corner, right_mouth_corner]
        """
        # Indices for relevant features based on dlib's 68 landmarks
        left_eye_indices = list(range(36, 42))     # Left eye
        right_eye_indices = list(range(42, 48))    # Right eye
        nose_tip_index = 30                        # Nose tip (index 30)
        left_mouth_corner_index = 48               # Left mouth corner
        right_mouth_corner_index = 54              # Right mouth corner

        # Calculate centers by averaging over the eye landmarks
        left_eye_center = shape_68[left_eye_indices].mean(axis=0)
        right_eye_center = shape_68[right_eye_indices].mean(axis=0)
        nose_tip = shape_68[nose_tip_index]
        left_mouth_corner = shape_68[left_mouth_corner_index]
        right_mouth_corner = shape_68[right_mouth_corner_index]

        # Compose the final (5, 2) array
        shape_5 = np.array([
            left_eye_center,
            right_eye_center,
            nose_tip,
            left_mouth_corner,
            right_mouth_corner
        ])

        return shape_5
