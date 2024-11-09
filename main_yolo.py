import os
import cv2
import random
import warnings
import argparse
import logging
import numpy as np

from ultralytics import YOLO
import onnxruntime
from typing import Union, List, Tuple
from models import SCRFD, ArcFace
from utils.helpers import compute_similarity, draw_bbox_info, draw_bbox
import serial
import time

# TODO: write a functions to send signal to ESP32 in other file follow class priciple

# Modified serial connection setup with better error handling
serial_enabled = False
ser = None

# def initialize_serial():
#     global serial_enabled, ser
#     try:
#         ser = serial.Serial('COM4', 115200, timeout=1)
#         time.sleep(2)  # Wait for ESP32 to start up
#         serial_enabled = True
#         print("Successfully connected to ESP32 on COM4")
#     except serial.SerialException as e:
#         if "PermissionError" in str(e):
#             print("Error: Cannot access COM4 - Port may be in use by another program")
#             print("Please close any other applications using COM4 and try again")
#         else:
#             print(f"Warning: Could not open serial port: {e}")
#         serial_enabled = False

# # Call initialize_serial at startup
# initialize_serial()

# def send_signal_to_esp32():
#     global ser
#     if serial_enabled and ser:
#         try:
#             ser.write(b'FACE_DETECTED\n')
#             print("Signal sent to ESP32")
#         except serial.SerialException as e:
#             print(f"Error sending signal: {e}")
#             # If we lose connection, try to re-initialize
#             initialize_serial()
#     else:
#         print("Serial communication is disabled - skipping signal")

# warnings.filterwarnings("ignore")



def parse_args():
    parser = argparse.ArgumentParser(description="Face Detection-and-Recognition")
    parser.add_argument(
        "--det-weight",
        type=str,
        default="./weights/yolov8n-face.pt",
        help="Path to detection model"
    )
    parser.add_argument(
        "--rec-weight",
        type=str,
        default="./weights/w600k_r50.onnx",
        help="Path to recognition model"
    )
    parser.add_argument(
        "--similarity-thresh",
        type=float,
        default=0.4,
        help="Similarity threshold between faces"
    )
    parser.add_argument(
        "--confidence-thresh",
        type=float,
        default=0.5,
        help="Confidence threshold for face detection"
    )
    parser.add_argument(
        "--faces-dir",
        type=str,
        default="./faces",
        help="Path to faces stored dir"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video file or video camera source. i.e 0 - webcam"
    )
    parser.add_argument(
        "--max-num",
        type=int,
        default=0,
        help="Maximum number of face detections from a frame"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level"
    )

    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), None),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def build_targets(detector, recognizer, params: argparse.Namespace) -> List[Tuple[np.ndarray, str]]:
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
    for filename in os.listdir(params.faces_dir):
        name = filename[:-4]
        image_path = os.path.join(params.faces_dir, filename)

        image = cv2.imread(image_path)
        
        # Force CPU inference
        detections = detector(image, device='cpu')

        bboxes = detections[0].boxes.xyxy.numpy()  # Remove .cpu() since we're already on CPU
        kpss = detections[0].keypoints.xy.numpy()  # Remove .cpu() since we're already on CPU
    
        if len(kpss) == 0:
            logging.warning(f"No face detected in {image_path}. Skipping...")
            continue

        embedding = recognizer(image, kpss[0])
        targets.append((embedding, name))

    return targets


def frame_processor(
    frame: np.ndarray,
    detector: YOLO,
    recognizer: ArcFace,
    targets: List[Tuple[np.ndarray, str]],
    colors: dict,
    params: argparse.Namespace
) -> np.ndarray:
    """
    Process a video frame for face detection and recognition.

    Args:
        frame (np.ndarray): The video frame.
        detector (YOLO): Face detector model.
        recognizer (ArcFace): Face recognizer model.
        targets (List[Tuple[np.ndarray, str]]): List of target feature vectors and names.
        colors (dict): Dictionary of colors for drawing bounding boxes.
        params (argparse.Namespace): Command line arguments.

    Returns:
        np.ndarray: The processed video frame.
    """
    # Force CPU inference
    detections = detector(frame, device='cpu')

    bboxes = detections[0].boxes.xyxy.numpy()
    kpss = detections[0].keypoints.xy.numpy()

    for bbox, kps in zip(bboxes, kpss):
        try:
            # Ensure keypoints are in correct format
            kps = np.array(kps, dtype=np.float32).reshape(5, 2)
            embedding = recognizer(frame, kps)

            max_similarity = 0
            best_match_name = "Unknown"
            for target, name in targets:
                similarity = compute_similarity(target, embedding)
                if similarity > max_similarity and similarity > params.similarity_thresh:
                    max_similarity = similarity
                    best_match_name = name

            # if best_match_name == "NguyenQuangLinh":
            #     send_signal_to_esp32()

            if best_match_name != "Unknown":
                color = colors[best_match_name]
                draw_bbox_info(frame, bbox, similarity=max_similarity, name=best_match_name, color=color)
            else:
                draw_bbox(frame, bbox, (255, 0, 0))
        except Exception as e:
            logging.error(f"Error processing face: {e}")
            continue

    return frame


def main(params):
    setup_logging(params.log_level)

    # Force CPU device
    detector = YOLO(params.det_weight)
    detector.to('cpu')
    recognizer = ArcFace(params.rec_weight)

    targets = build_targets(detector, recognizer, params)
    colors = {name: (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)) for _, name in targets}

    cap = cv2.VideoCapture(params.source)
    if not cap.isOpened():
        raise Exception("Could not open video or webcam")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = frame_processor(frame, detector, recognizer, targets, colors, params)
        out.write(frame)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    if args.source.isdigit():
        args.source = int(args.source)
    main(args)
