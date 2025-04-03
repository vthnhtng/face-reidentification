from interface.detector import Detector as DetectorInterface
from interface.recognizer import Recognizer as RecognizerInterface
from interface.vector_storage import VectorStorage as VectorStorageInterface
from typing import Union, List, Tuple
from utils.helpers import compute_similarity, draw_bbox_info, draw_bbox
from models import ArcFace
import numpy as np
import time
from typing import Tuple
import numpy as np

class ModelService:
    COLOR = {
        "known": (0, 255, 0),
        "unknown": (255, 0, 0),
    }

    def __init__(
        self,
        detector: DetectorInterface,
        recognizer: RecognizerInterface,
        vector_storage: VectorStorageInterface
    ):
        """
        Initialize the FaceReIdentifier with a detector, recognizer, and vector storage.
        
        Args:
            detector (DetectorInterface): The face detector instance.
            recognizer (RecognizerInterface): The face recognizer instance.
            vector_storage (VectorStorageInterface): The vector storage instance.
        """
        self.detector = detector
        self.recognizer = recognizer
        self.vector_storage = vector_storage

    def add_new_user(self, image: np.ndarray, user_data: dict) -> str: 
        """
        Add a new user to the re-identification system.
        
        Args:
            image (np.ndarray): Input image of the user.
            user_data (dict): Metadata associated with the user.
            
        Returns:
            str: Unique identifier for the added user.
        """
        bounding_box, keypoint = self.detector.detect_face(image)
        if bounding_box is None or bounding_box.size == 0 or keypoint is None or keypoint.size == 0:
            raise ValueError("No face detected or keypoints are empty.")
        
        embedding = self.recognizer.recognize_face(image, keypoint)
        vector_id = self.vector_storage.add_vector(
            vector=embedding,
            metadata=user_data
        )

        return vector_id

    def frame_process(
            self,
            frame: np.ndarray,
        ) -> Tuple[np.ndarray, str, float]:
        """
        Process a video frame for face detection and recognition, focusing on a single face
        with the highest similarity.

        Caches the embedding and vector search results to run the expensive operations only once per second.

        Returns:
            Tuple[np.ndarray, str, float]: A tuple containing:
                - The processed video frame with face detection visualizations
                - Best match name
                - Max similarity
        """
        # Detect face and keypoints.
        bounding_box, keypoint = self.detector.detect_face(frame)
        if len(bounding_box) == 0 or len(keypoint) == 0:
            return frame

        # Initialize cache attributes if they don't exist.
        if not hasattr(self, '_last_update_time'):
            self._last_update_time = 0
            self._cached_embedding = None
            self._cached_search_results = None

        current_time = time.time()
        # Update cached results once per second.
        if current_time - self._last_update_time >= 1:
            self._cached_embedding = self.recognizer.recognize_face(frame, keypoint)
            self._cached_search_results = self.vector_storage.search(self._cached_embedding)
            self._last_update_time = current_time

        embedding = self._cached_embedding
        search_results = self._cached_search_results

        # No search results found; mark as unknown.
        if len(search_results) == 0:
            draw_bbox(frame, bounding_box, self.COLOR["unknown"])
            return frame

        # Use the best match from the search results.
        best_match = search_results[0]
        similarity = best_match['score']
        username = best_match['metadata']['username']

        draw_bbox_info(
            frame=frame,
            bbox=bounding_box,
            similarity=similarity,
            name=username,
            color=self.COLOR["known"]
        )

        return frame
