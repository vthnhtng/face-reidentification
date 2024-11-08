import yaml
from ultralytics import YOLO
from arcface import ArcFace

with open('params.yaml', 'r') as f:
    config = yaml.safe_load(f)

class Server:
    def __init__(self):
        # Detection model
        self.detection_model_weight = config['detection_model']['weight']
        self.detection_model_confidence_threshold = config['detection_model']['confidence_threshold']
        self.detection_model_face_per_frame = config['detection_model']['face_per_frame']

        # Recognition model
        self.recognition_model_weight = config['recognition_model']['weight']
        self.recognition_model_similarity_threshold = config['recognition_model']['similarity_threshold']

    def load_models(self):
        self.detection_model = YOLO(self.detection_model_weight)
        self.recognition_model = ArcFace(self.recognition_model_weight)
    
