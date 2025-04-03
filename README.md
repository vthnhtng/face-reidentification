# Face Re-Identification system by YOLOv8 and ArcFace with simple web ui for new face registration
## Original Repository

This project is based on the [face-reidentification repository](https://github.com/yakhyo/face-reidentification) by [yakhyo](https://github.com/yakhyo).

## Features

1. **Web Interface**: A simple and intuitive web UI for managing the system.
2. **Face Registration**: Easily register new faces directly through the web interface.
3. **Face Re-Identification**: Perform face re-identification using advanced detection and recognition techniques.

## Tech Stack

1. **YOLOv8**: Utilized for accurate and efficient face detection.
2. **ArcFace**: Generates embeddings for face recognition.
3. **Qdrant Vector Database**: Stores embeddings and retrieves similar vectors for re-identification.

## Requirements

- Python 3.8 or higher
- PyTorch
- OpenCV
- Flask
- Additional dependencies listed in `requirements.txt`

## Acknowledgments

- Original repository by [yakhyo](https://github.com/yakhyo/face-reidentification)
- [YOLOv8](https://github.com/ultralytics/ultralytics) by Ultralytics
- ArcFace implementation
- Model weights are available in the original repository

## How to Run

1. Install Python 3.10.9 or a compatible version.
2. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
3. Start the Qdrant container:
  ```bash
  docker run -p 6333:6333 -p 6334:6334 \
     -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
     qdrant/qdrant
  ```
4. Launch the application:
  ```bash
  python app.py
  ```