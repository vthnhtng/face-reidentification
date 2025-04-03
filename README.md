# Face Re-Identification system by YOLOv8 and ArcFace with simple web ui for new face registration
## The origin repository is from https://github.com/yakhyo/face-reidentification

1. Provide simple the web UI
2. Register new faces through the web interface
3. Face re-identification

## Tech Stack
1. YOLOv8 for face detection
2. ArcFace for face recognition(generate embeddings)
3. Qdrant vector database for vector storage and similar vectors retrieval

## Requirements
- Python 3.8+
- PyTorch
- OpenCV
- Flask
- Other dependencies listed in requirements.txt

## Acknowledgments
- Original work by [yakhyo](https://github.com/yakhyo/face-reidentification)
- YOLOv8 by Ultralytics
- ArcFace implementation
- You can find model weights in the origin repo