import cv2
import numpy as np
import os

class FaceReidentifier():
    def __init__(self, detector, recognizer, embedding_storage):
        self.detector = detector
        self.recognizer = recognizer
        self.embedding_storage = embedding_storage

    def get_embedding(self, image: cv2.typing.MatLike) -> np.ndarray:
        """
        Build and save the embedding for a new user using face detection and recognition.

        Args:
            image (cv2.typing.MatLike): The input image containing the face of the user.
            user_id (int): The unique identifier for the user.

        Raises:
            ValueError: no face is detected or embedding generation fails.

        Returns:
            np.ndarray: The generated embedding for the detected face.
        """

        # Create faces directory if it doesn't exist
        detections = self.detector(image, device='cpu', verbose=False)

        kpss = detections[0].keypoints.xy.cpu().numpy()

        if len(kpss) == 0:
            raise ValueError(f"No face is detected in the image")

        embedding = self.recognizer(image, kpss[0])

        return embedding

    def save_embedding(self, embedding: np.ndarray, user_id: int) -> bool:
        try:
            self.embedding_storage.add_embedding(embedding)
            return True
        except Exception as e:
            print(f"Error saving embedding: {str(e)}")
            return False

    def search_similar_embedding(self, query_embedding: np.ndarray, threshold: float, top_k=1) -> tuple:
        try:
            distances, indices = self.embedding_storage.search(query_embedding, k=top_k)
            
            # Filter results based on the threshold
            filtered_results = [
                (index, distance)
                for index, distance in zip(indices[0], distances[0])
                if index != -1 and distance <= threshold
            ]

            return filtered_results
        except Exception as e:
            return []


    def frame_process(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Process a video frame for face detection and recognition.
        Returns:
            Tuple[np.ndarray, bool]: A tuple containing:
                - The processed video frame with face detection visualizations
                - A boolean indicating if a known face was detected
        """
        detections = self.detector(frame, device="cpu", verbose=False)

        bboxes = detections[0].boxes.xyxy.cpu().numpy()
        kpss = detections[0].keypoints.xy.cpu().numpy()

        face_detected = False

        for bbox, kps in zip(bboxes, kpss):
            embedding = self.recognizer(frame, kps)

            max_similarity = 0
            best_match_name = ""

            for target, name in self.targets:
                similarity = compute_similarity(target, embedding) # query embedding vs target embedding
                if similarity > max_similarity and similarity > self.RECOGNIZER_SIMILARITY_THRESHOLD:
                    max_similarity = similarity
                    best_match_name = name

            if best_match_name == "":
                draw_bbox(frame, bbox, self.COLOR["unknown"])
            else:
                draw_bbox_info(frame, bbox, similarity=max_similarity, name=best_match_name, color=self.COLOR["known"])
                face_detected = True

        return frame, face_detected
