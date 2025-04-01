import cv2
import numpy as np
import tensorflow as tf
import dlib


class YawnDetector:
    def __init__(
        self,
        landmark_path="shape_predictor_68_face_landmarks.dat",
        model_path="no_augmented/yawn_detection_model.keras",
        class_indices_path="no_augmented/class_indices.npy",
    ):
        """Initialize with paths to model files"""
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(landmark_path)
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.class_indices = np.load(class_indices_path, allow_pickle=True).item()
        self.reverse_class_indices = {v: k for k, v in self.class_indices.items()}

    def process_image(self, image_path):
        """Complete pipeline for single image processing"""
        # Step 1: Detect and crop mouth
        mouth_crop = self._crop_mouth_region(image_path)
        if mouth_crop is None:
            return {"error": "No face/mouth detected"}

        # Step 2: Preprocess for model
        processed_img = self._preprocess_mouth_crop(mouth_crop)
        if processed_img is None:
            return {"error": "Image preprocessing failed"}

        # Step 3: Make prediction
        return self._predict_yawn(processed_img)

    def _crop_mouth_region(self, image_path, padding=20):
        """Internal: Detect and crop mouth region"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading image: {image_path}")
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if not faces:
            print("No face detected")
            return None

        landmarks = self.predictor(gray, faces[0])
        mouth_points = np.array(
            [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
        )
        x, y, w, h = cv2.boundingRect(mouth_points)

        # Apply padding with boundary checks
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)

        return image[y : y + h, x : x + w]

    def _preprocess_mouth_crop(self, mouth_crop):
        """Internal: Preprocess cropped mouth image"""
        try:
            # Convert to tensor and apply model-specific preprocessing
            img = tf.convert_to_tensor(mouth_crop)
            img = tf.image.resize(img, (224, 224))
            img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
            return tf.expand_dims(img, 0)
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None

    def _predict_yawn(self, processed_img):
        """Internal: Make prediction on preprocessed image"""
        try:
            pred = self.model.predict(processed_img, verbose=0)[0]
            class_idx = np.argmax(pred)
            confidence = float(pred[class_idx])
            class_name = self.reverse_class_indices[class_idx]

            return {
                "class": class_name,
                "confidence": f"{confidence:.2%}",
                "all_scores": {
                    k: f"{v:.2%}" for k, v in zip(self.class_indices.keys(), pred)
                },
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            return {"error": str(e)}
