import os
import cv2
import numpy as np
from PIL import Image
from datetime import datetime


class FaceDetector:
    def __init__(self):
        """
        Initialize FaceDetector with both Haar Cascade and DNN face detection.
        """
        # Initialize traditional Haar Cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Initialize DNN face detector
        self.dnn_face_detector = self._load_dnn_face_detector()

    def _load_dnn_face_detector(self):
        """Load the DNN face detector model"""
        prototxt_path = "models/deploy.prototxt"
        weights_path = "models/res10_300x300_ssd_iter_140000.caffemodel"

        if not os.path.exists(prototxt_path) or not os.path.exists(weights_path):
            print(
                "DNN face detector model files not found. Falling back to Haar Cascade only."
            )
            return None

        return cv2.dnn.readNet(prototxt_path, weights_path)

    def detect_faces(self, image):
        """
        Detect faces in the given image using both Haar Cascade and DNN (if available).

        :param image: PIL Image or numpy array
        :return: Dictionary with face detection results
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

        # First try DNN face detection if available
        if self.dnn_face_detector is not None:
            dnn_results = self._detect_faces_dnn(image)
            if dnn_results["total_faces"] > 0:
                return dnn_results

        # Fall back to Haar Cascade if DNN not available or didn't find faces
        return self._detect_faces_haar(image)

    def _detect_faces_haar(self, image):
        """Detect faces using Haar Cascade"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        # Convert from (x, y, w, h) to (top, right, bottom, left)
        face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in faces]

        return self._prepare_face_results(face_locations)

    def _detect_faces_dnn(self, image, confidence_threshold=0.5):
        """Detect faces using DNN model"""
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

        self.dnn_face_detector.setInput(blob)
        detections = self.dnn_face_detector.forward()

        face_locations = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Ensure the bounding boxes fall within the image dimensions
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # Convert to (top, right, bottom, left) format
                face_locations.append((startY, endX, endY, startX))

        return self._prepare_face_results(face_locations)

    def _prepare_face_results(self, face_locations):
        """Prepare face detection results"""
        results = {
            "total_faces": len(face_locations),
            "faces": [],
        }

        for i, location in enumerate(face_locations):
            face_result = {
                "id": i + 1,
                "location": location,
            }
            results["faces"].append(face_result)

        return results

    def save_face(self, image, face_location, save_dir, prefix="face"):
        """
        Save a cropped face from the image.

        :param image: PIL Image or numpy array
        :param face_location: Tuple of face location (top, right, bottom, left)
        :param save_dir: Directory to save the face
        :param prefix: Prefix for the filename
        :return: Path to the saved face image
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

        # Unpack face location
        top, right, bottom, left = face_location

        # Add some padding
        padding = 30
        top = max(0, top - padding)
        left = max(0, left - padding)
        bottom = min(image.shape[0], bottom + padding)
        right = min(image.shape[1], right + padding)

        # Crop the face
        cropped_face = image[top:bottom, left:right]

        # Convert to PIL Image for saving
        cropped_face_pil = Image.fromarray(
            cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        )

        # Generate filename
        filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(save_dir, filename)

        # Save the cropped face
        cropped_face_pil.save(filepath)

        return filepath
