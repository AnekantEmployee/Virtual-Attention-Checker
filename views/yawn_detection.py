import cv2
import dlib
import numpy as np
from PIL import Image
import tensorflow as tf


class YawnDetector:
    def __init__(
        self,
        landmark_path="models/yawn/shape_predictor_68_face_landmarks.dat",
        model_path="models/yawn/yawn_detection_model.keras",
        class_indices_path="models/yawn/class_indices.npy",
    ):
        """Initialize with paths to model files"""
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(landmark_path)
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.class_indices = np.load(class_indices_path, allow_pickle=True).item()
        self.reverse_class_indices = {v: k for k, v in self.class_indices.items()}

    def process_image(self, image_input):
        image_np = self._convert_to_numpy(image_input)
        if image_np is None:
            return {"error": "Invalid image input", "is_verified": False}

        mouth_crop = self._crop_mouth_region(image_np)
        if mouth_crop is None:
            return {"error": "No face/mouth detected", "is_verified": False}

        processed_img = self._preprocess_mouth_crop(mouth_crop)
        if processed_img is None:
            return {"error": "Image preprocessing failed", "is_verified": False}

        return self._predict_yawn(processed_img)

    def _convert_to_numpy(self, image_input):
        """Convert various input types to numpy array (BGR format)
        Returns:
            numpy.ndarray: BGR image array, or None if conversion fails
        """
        if isinstance(image_input, str):
            # Input is file path
            img = cv2.imread(image_input)
            if img is None:
                print(f"Error reading image: {image_input}")
                return None
            return img

        elif isinstance(image_input, Image.Image):
            # Input is PIL Image
            try:
                img_rgb = np.array(image_input)
                # Handle different image modes
                if img_rgb.ndim == 2:  # Grayscale
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2BGR)
                elif img_rgb.shape[2] == 4:  # RGBA
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2BGR)
                else:  # RGB
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                return img_bgr
            except Exception as e:
                print(f"Error converting PIL image: {e}")
                return None

        elif isinstance(image_input, np.ndarray):
            # Input is numpy array
            try:
                if len(image_input.shape) == 2:  # Grayscale
                    return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
                elif image_input.shape[2] == 4:  # RGBA
                    return cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR)
                elif image_input.shape[2] == 3:  # Assume RGB
                    return cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
                return image_input
            except Exception as e:
                print(f"Error converting numpy array: {e}")
                return None

        print(f"Unsupported image input type: {type(image_input)}")
        return None

    def _crop_mouth_region(self, image_np, padding=20):
        """Detect and crop mouth region from numpy array (BGR format)
        Returns:
            numpy.ndarray: Cropped mouth region, or None if detection fails
        """
        if image_np is None:
            return None

        try:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 1)

            if not faces:
                print("No face detected")
                # return None
                return image_np

            landmarks = self.predictor(gray, faces[0])
            mouth_points = np.array(
                [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
            )
            x, y, w, h = cv2.boundingRect(mouth_points)

            # Apply padding with boundary checks
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image_np.shape[1] - x, w + 2 * padding)
            h = min(image_np.shape[0] - y, h + 2 * padding)

            return image_np[y : y + h, x : x + w]
        except Exception as e:
            print(f"Error cropping mouth region: {e}")
            return None

    def _preprocess_mouth_crop(self, mouth_crop):
        """Preprocess cropped mouth image for model input"""
        if mouth_crop is None:
            return None

        try:
            # Convert BGR to RGB
            mouth_rgb = cv2.cvtColor(mouth_crop, cv2.COLOR_BGR2RGB)

            # Convert to tensor and apply model-specific preprocessing
            img = tf.convert_to_tensor(mouth_rgb)
            img = tf.image.resize(img, (224, 224))
            img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
            return tf.expand_dims(img, 0)
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None

    def _predict_yawn(self, processed_img):
        """Make prediction on preprocessed image"""

        if processed_img is None:
            return {"error": "Invalid input for prediction", "is_verified": False}

        try:
            pred = self.model.predict(processed_img, verbose=0)[0]
            class_idx = np.argmax(pred)
            class_name = self.reverse_class_indices[class_idx]

            return {"yawn_state": 0 if class_name == "yawn" else 1, "is_verified": True}

        except Exception as e:
            print(f"Prediction error: {e}")
            return {"error": str(e), "is_verified": False}
