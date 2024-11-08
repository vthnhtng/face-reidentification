import yaml

try:
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    # Detection model params
    DETECTION_MODEL_WEIGHT = params['detection_model']['weight']
    DETECTION_MODEL_CONFIDENCE_THRESHOLD = params['detection_model']['confidence_threshold'] 
    DETECTION_MODEL_FACE_PER_FRAME = params['detection_model']['face_per_frame']

    # Recognition model params
    RECOGNITION_MODEL_WEIGHT = params['recognition_model']['weight']
    RECOGNITION_MODEL_SIMILARITY_THRESHOLD = params['recognition_model']['similarity_threshold']

    # params
    IMAGE_EXTENSION = params['params']['image_extension']
    FACES_DIR = params['params']['faces_dir']

except Exception as e:
    raise Exception(f"Failed to load params file: {str(e)}")
