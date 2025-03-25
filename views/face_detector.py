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
        # Initialize traditional Haar Cascade with better parameters
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Initialize DNN face detector
        self.dnn_face_detector = self._load_dnn_face_detector()
        self.dnn_confidence_threshold = (
            0.7  # Increased from 0.5 to reduce false positives
        )

    def _load_dnn_face_detector(self):
        """Load the DNN face detector model"""
        prototxt_path = "models/deploy.prototxt"
        weights_path = "models/res10_300x300_ssd_iter_140000.caffemodel"

        if not os.path.exists(prototxt_path) or not os.path.exists(weights_path):
            print(
                "DNN face detector model files not found. Falling back to Haar Cascade only."
            )
            return None

        net = cv2.dnn.readNet(prototxt_path, weights_path)
        net.setPreferableBackend(
            cv2.dnn.DNN_BACKEND_CUDA
            if cv2.cuda.getCudaEnabledDeviceCount()
            else cv2.dnn.DNN_BACKEND_OPENCV
        )
        net.setPreferableTarget(
            cv2.dnn.DNN_TARGET_CUDA
            if cv2.cuda.getCudaEnabledDeviceCount()
            else cv2.dnn.DNN_TARGET_CPU
        )
        return net

    def detect_faces(self, image):
        """
        Detect faces in the given image using both Haar Cascade and DNN (if available).
        Returns the most confident detections.

        :param image: PIL (Python image library) Image or numpy array
        :return: Dictionary with face detection results
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

        # Get detections from both methods
        dnn_results = self._detect_faces_dnn(image) if self.dnn_face_detector else None
        haar_results = self._detect_faces_haar(image)

        # If DNN found faces, use those (they're generally more accurate)
        if dnn_results and dnn_results["total_faces"] > 0:
            return dnn_results

        # Otherwise use Haar results
        return haar_results

    def _detect_faces_haar(self, image):
        """Detect faces using Haar Cascade with tuned parameters"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Tuned parameters for better accuracy
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # Reduced from 1.1 to be more precise
            minNeighbors=7,  # Increased from 5 to reduce false positives
            minSize=(50, 50),  # Increased minimum size
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # Convert from (x, y, w, h) to (top, right, bottom, left)
        face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in faces]

        return self._prepare_face_results(face_locations)

    def _detect_faces_dnn(self, image):
        """Detect faces using DNN model with higher confidence threshold"""
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image,
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
            swapRB=False,  # Already in BGR format
            crop=False,
        )

        self.dnn_face_detector.setInput(blob)
        detections = self.dnn_face_detector.forward()

        face_locations = []
        confidences = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.dnn_confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Ensure the bounding boxes fall within the image dimensions
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w - 1, endX), min(h - 1, endY)

                # Only keep if the face is reasonably sized
                face_width = endX - startX
                face_height = endY - startY
                min_face_size = (
                    min(image.shape[0], image.shape[1]) * 0.1
                )  # At least 10% of smaller dimension

                if face_width > min_face_size and face_height > min_face_size:
                    # Convert to (top, right, bottom, left) format
                    face_locations.append((startY, endX, endY, startX))
                    confidences.append(confidence)

        # Sort faces by confidence (highest first)
        if confidences:
            face_locations = [
                face
                for _, face in sorted(zip(confidences, face_locations), reverse=True)
            ]

        return self._prepare_face_results(face_locations)

    def _prepare_face_results(self, face_locations):
        """Prepare face detection results with additional validation"""
        validated_locations = []

        for location in face_locations:
            top, right, bottom, left = location
            # Validate the face dimensions are reasonable
            height = bottom - top
            width = right - left
            aspect_ratio = width / height

            # Typical face aspect ratio is between 0.7 and 1.5
            if 0.6 <= aspect_ratio <= 1.6:
                validated_locations.append(location)

        results = {
            "total_faces": len(validated_locations),
            "faces": [],
        }

        for i, location in enumerate(validated_locations):
            face_result = {
                "id": i + 1,
                "location": location,
                "confidence": 1.0,  # Placeholder, actual confidence from DNN would be better
            }
            results["faces"].append(face_result)

        return results

    def save_face(self, image, face_location, save_dir, prefix="face"):
        """
        Save a cropped face from the image with improved validation.

        :param image: PIL Image or numpy array
        :param face_location: Tuple of face location (top, right, bottom, left)
        :param save_dir: Directory to save the face
        :param prefix: Prefix for the filename
        :return: Path to the saved face image or None if invalid
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

        # Unpack face location
        top, right, bottom, left = face_location

        # Validate face dimensions
        height = bottom - top
        width = right - left
        if height <= 0 or width <= 0:
            return None

        # Add proportional padding (20% of face width/height)
        h_pad = int(width * 0.2)
        v_pad = int(height * 0.2)

        top = max(0, top - v_pad)
        left = max(0, left - h_pad)
        bottom = min(image.shape[0], bottom + v_pad)
        right = min(image.shape[1], right + h_pad)

        # Validate final dimensions
        if bottom <= top or right <= left:
            return None

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
