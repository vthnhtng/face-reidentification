import params_loader
from ultralytics import YOLO
from models import ArcFace
import numpy as np
from typing import Union, List, Tuple
from utils.helpers import compute_similarity, draw_bbox_info, draw_bbox
import os
import cv2

class ModelServices:
    # Detection model params
    DETECTOR_WEIGHT = params_loader.DETECTION_MODEL_WEIGHT
    DETECTOR_CONFIDENCE_THRESHOLD = params_loader.DETECTION_MODEL_CONFIDENCE_THRESHOLD
    DETECTOR_FACE_PER_FRAME = params_loader.DETECTION_MODEL_FACE_PER_FRAME

    # Recognition model params
    RECOGNIZER_WEIGHT = params_loader.RECOGNITION_MODEL_WEIGHT
    RECOGNIZER_SIMILARITY_THRESHOLD = params_loader.RECOGNITION_MODEL_SIMILARITY_THRESHOLD

    # params
    COLOR = {
        "known": (0, 255, 0),
        "unknown": (255, 0, 0),
    }
    IMAGE_EXTENSION = params_loader.IMAGE_EXTENSION
    FACES_DIR = params_loader.FACES_DIR


    def __init__(self):
        self.detector = YOLO(self.DETECTOR_WEIGHT)
        self.recognizer = ArcFace(self.RECOGNIZER_WEIGHT)
        self.targets = self.build_targets()


    def build_targets(self):
        """
        Build targets using face detection and recognition.

        Args:
            detector (YOLO): Face detector model.
            recognizer (ArcFaceONNX): Face recognizer model.
            params (argparse.Namespace): Command line arguments.

        Returns:
            List[Tuple[np.ndarray, str]]: A list of tuples containing feature vectors and corresponding image names.
        """
        targets = []
        for filename in os.listdir(self.FACES_DIR):
            name = filename.split(".")[0]
            image_path = os.path.join(self.FACES_DIR, filename)

            image = cv2.imread(image_path)
            detections = self.detector(image)

            bboxes = detections[0].boxes.xyxy.cpu().numpy()
            kpss = detections[0].keypoints.xy.cpu().numpy()

            if len(kpss) == 0:
                print(f"No face detected in {image_path}. Skipping...")
                continue
                
            for face_idx in range(len(kpss)):
                if face_idx >= self.DETECTOR_FACE_PER_FRAME:
                    break

                embedding = self.recognizer(image, kpss[face_idx])
                targets.append((embedding, name))

        return targets

    def frame_process(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Process a video frame for face detection and recognition.

        Args:
            frame (np.ndarray): The video frame.
            detector (YOLO): Face detector model.
            recognizer (ArcFaceONNX): Face recognizer model.

        Returns:
            np.ndarray: The processed video frame.
        """
        detections = self.detector(frame)

        bboxes = detections[0].boxes.xyxy.cpu().numpy()
        kpss = detections[0].keypoints.xy.cpu().numpy()

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
            

        return frame
