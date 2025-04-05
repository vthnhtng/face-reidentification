from detector.dlib import Dlib
from detector.yolo import Yolo
from recognizer.arcface import ArcFace
from vector_storage.qdrant import Qdrant
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from model_service import ModelService
import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO
from flask import Response
import time
import os
from PIL import Image
import cv2
from dotenv import load_dotenv

load_dotenv()

# Detector env
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS")
CONFIDENCE_THRESHOULD = float(os.getenv("CONFIDENCE_THRESHOLD"))

# Recognizer env
ARCFACE_WEIGHTS = os.getenv("ARCFACE_WEIGHTS")

# Vector database env
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME=os.getenv("COLLECTION_NAME")
SIMILARITY_THRESHOLD=float(os.getenv("SIMILARITY_THRESHOLD"))

dlib = Dlib("shape_predictor_68_face_landmarks.dat", 0.1)
yolo = Yolo(YOLO_WEIGHTS, 0.5)
arcface = ArcFace(ARCFACE_WEIGHTS)
qdrant = Qdrant(QDRANT_URL, COLLECTION_NAME, SIMILARITY_THRESHOLD)

model_service = ModelService(
    detector=dlib,
    recognizer=arcface,
    vector_storage=qdrant
)

filepath = 'faces/TungVT/1.png'
model_service.add_new_user(
    cv2.imread(filepath),
    {
        'username': "TungVT",
        'image': filepath
    }
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    try:
        processed_frame = model_service.frame_process(frame)
        cv2.imshow('Face Recognition', processed_frame)
    except Exception as e:
        cv2.imshow('Face Recognition', frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
