import os
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import mediapipe as mp


class FaceDetector:
    def __init__(self):
        """
        Initialize FaceDetector with MediaPipe face detection.
        """
        # Initialize MediaPipe Face Detection
        mp_face_detection = mp.solutions.face_detection
        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for short-range, 1 for full-range
            min_detection_confidence=0.5,
        )

    def detect_faces(self, image):
        """
        Detect faces in the given image using MediaPipe.

        :param image: PIL Image or numpy array
        :return: Dictionary with face detection results
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

        # Detect faces using MediaPipe
        return self._detect_faces_mediapipe(image)

    def _detect_faces_mediapipe(self, image):
        """Detect faces using MediaPipe"""
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)

        face_locations = []

        if results.detections:
            for detection in results.detections:
                # Get bounding box
                box = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape

                # Convert relative coordinates to absolute coordinates
                x = int(box.xmin * iw)
                y = int(box.ymin * ih)
                w = int(box.width * iw)
                h = int(box.height * ih)

                # Convert to (top, right, bottom, left) format
                top = y
                right = x + w
                bottom = y + h
                left = x

                face_locations.append((top, right, bottom, left))

        # os.remove(image_path)
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
