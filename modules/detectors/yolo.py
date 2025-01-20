from ultralytics import YOLO

__all__ = ["Yolo"]

class Yolo:
    def __init__(self, yolo_weight: str) -> None:
        """
        Initialize the YOLO face detector with a pre-trained model.
        
        Args:
            yolo_weight (str): Path to the pre-trained YOLO model.
        """

        self.model = YOLO(yolo_weight)

    def detect(self, image, **kwargs) -> tuple:
        """
        Detect faces in the image using the YOLO model.

        Args:
            image: The input image to process (numpy array, PIL image, etc.).
            **kwargs: Optional parameters for the detection (not used here, but can be extended).

        Returns:
            list: A list containing two elements:
                - Bounding boxes (xyxy format) for detected faces.
                - Keypoints for detected faces (if available).
        """

        detections = self.model(
            image,
            device=kwargs.get('device', 'cpu'),
            verbose=kwargs.get('verbose', False)
        )

        bboxes = detections[0].boxes.xyxy.cpu().numpy()
        kpss = detections[0].keypoints.xy.cpu().numpy()

        return bboxes, kpss
