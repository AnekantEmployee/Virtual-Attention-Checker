import cv2
import numpy as np
from ultralytics import YOLO


class EyeDetector:
    def __init__(
        self,
        model_path="models/eye/best.pt",
    ):
        self.model_path = model_path
        self.model = YOLO(self.model_path)

    def get_eye_state(self, image_pil):
        """Process a single PIL image and save results to CSV."""

        # Convert PIL Image to numpy array
        image = np.array(image_pil)
        # Convert RGB to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Run inference
        results = self.model.predict(
            source=image, conf=0.5, device="cpu", verbose=False
        )

        best_eye_state = -1
        best_confidence = 0.0

        for r in results:
            for box in r.boxes:
                confidence = float(box.conf)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_eye_state = int(box.cls)

        if best_eye_state != -1:
            return {"eye_state": best_eye_state, "is_verified": True}
        else:
            return {"eye_state": -1, "is_verified": False}
