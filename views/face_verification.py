import os
import json
import cv2
import mtcnn
import torch
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from facenet_pytorch import InceptionResnetV1
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import time
from datetime import datetime
import shutil


class FaceVerification:
    def __init__(self):
        # Initialize face detector with min_face_size parameter
        self.face_detector = mtcnn.MTCNN()

        # Initialize FaceNet for face embeddings
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval()

        # Create directories for verified faces
        self.faces_dir = "faces"
        self.faces_json_path = os.path.join(self.faces_dir, "verified_faces.json")

        # Ensure directories exist
        os.makedirs(self.faces_dir, exist_ok=True)

        # Initialize JSON file if it doesn't exist
        if not os.path.exists(self.faces_json_path):
            with open(self.faces_json_path, "w") as f:
                json.dump([], f)

        # Load target images and train model
        target_dir = "target_img"
        self.targets = self.load_target_images(target_dir)
        if not self.targets:
            raise ValueError("No valid target images found in target directory")

        # Train model
        self.train_model()

    def is_valid_image(self, image_path, min_size=64):
        """Check if image exists and meets minimum size requirements"""
        try:
            if not os.path.exists(image_path):
                return False, "File does not exist"

            with Image.open(image_path) as img:
                width, height = img.size
                if width < min_size or height < min_size:
                    return False, f"Image too small ({width}x{height})"
                return True, "Valid image"
        except Exception as e:
            return False, f"Invalid image: {str(e)}"

    def load_target_images(self, target_dir):
        """Load all valid target images from directory"""
        targets = []
        for filename in os.listdir(target_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(target_dir, filename)
                valid, msg = self.is_valid_image(path)
                if valid:
                    targets.append(
                        {"name": os.path.splitext(filename)[0], "path": path}
                    )
                else:
                    print(f"Skipping target {filename}: {msg}")
        return targets

    def get_face_embedding(self, face_image):
        """Convert face image to embedding using FaceNet"""
        try:
            # Convert to tensor and preprocess
            face = cv2.resize(face_image, (160, 160))
            face = face.astype("float32")

            # Convert from BGR to RGB if needed
            if len(face.shape) == 3 and face.shape[2] == 3:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # Standardize pixel values
            mean, std = face.mean(), face.std()
            face = (face - mean) / std

            # Convert to PyTorch tensor
            face = torch.FloatTensor(face).permute(2, 0, 1).unsqueeze(0)

            # Get embedding
            with torch.no_grad():
                embedding = self.resnet(face)

            return embedding.numpy().flatten()
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return None

    def train_model(self):
        """Train model with target images using face embeddings"""
        face_data = []
        labels = []

        for target in self.targets:
            try:
                # Read image
                img = cv2.imread(target["path"])
                if img is None:
                    print(f"Could not read image: {target['path']}")
                    continue

                # Convert to RGB (MTCNN expects RGB)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Detect face with MTCNN
                results = self.face_detector.detect_faces(img_rgb)
                if len(results) == 0:
                    print(f"No face detected in target image: {target['path']}")
                    continue

                # Take the face with highest confidence
                best_face = max(results, key=lambda x: x["confidence"])

                # Extract face
                x, y, w, h = best_face["box"]
                # Ensure coordinates are within image bounds
                x, y = max(0, x), max(0, y)
                w, h = min(w, img.shape[1] - x), min(h, img.shape[0] - y)

                if w <= 0 or h <= 0:
                    print(f"Invalid face dimensions in {target['path']}")
                    continue

                face_img = img_rgb[y : y + h, x : x + w]

                # Get face embedding
                embedding = self.get_face_embedding(face_img)
                if embedding is None:
                    print(f"Could not generate embedding for {target['path']}")
                    continue

                face_data.append(embedding)
                labels.append(target["name"])

                print(f"Successfully processed target: {target['name']}")

            except Exception as e:
                print(f"Error processing target image {target['path']}: {str(e)}")
                continue

        if len(face_data) == 0:
            raise ValueError("No valid face data found in target images")

        # Encode labels
        self.le = LabelEncoder()
        encoded_labels = self.le.fit_transform(labels)

        # Use SVM classifier
        self.classifier = SVC(kernel="linear", probability=True)
        self.classifier.fit(face_data, encoded_labels)

        # Store the face data and labels for similarity calculation
        self.training_data = np.array(face_data)
        self.training_labels = np.array(labels)

    def verifyFace(self, screenshot):
        """Verify faces in screenshots using our improved model"""
        try:
            if not screenshot["face_data"]:
                return

            result = []
            current_time = datetime.now().isoformat()
            screenshot_id = screenshot.get(
                "id", str(int(time.time()))
            )  # Use provided ID or timestamp

            for face in screenshot["face_data"]:
                cropped_face_path = face["cropped_face_path"]

                # Check if face image is valid
                valid, msg = self.is_valid_image(cropped_face_path)
                if not valid:
                    print(f"Skipping {cropped_face_path}: {msg}")
                    return

                try:
                    # Process the face image
                    img = cv2.imread(cropped_face_path)
                    if img is None:
                        print(f"Could not read image: {cropped_face_path}")
                        continue

                    # Convert to RGB
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Detect face with MTCNN
                    face_results = self.face_detector.detect_faces(img_rgb)
                    if len(face_results) == 0:
                        print(f"No face detected in {cropped_face_path}")
                        continue

                    # Take the best face
                    best_face = max(face_results, key=lambda x: x["confidence"])
                    x, y, w, h = best_face["box"]

                    # Ensure coordinates are within bounds
                    x, y = max(0, x), max(0, y)
                    w, h = min(w, img.shape[1] - x), min(h, img.shape[0] - y)

                    if w <= 0 or h <= 0:
                        print(f"Invalid face dimensions in {cropped_face_path}")
                        continue

                    face_img = img_rgb[y : y + h, x : x + w]

                    # Get embedding
                    embedding = self.get_face_embedding(face_img)
                    if embedding is None:
                        print(f"Could not generate embedding for {cropped_face_path}")
                        continue

                    embedding = embedding.reshape(1, -1)

                    # Predict using SVM
                    predictions = self.classifier.predict_proba(embedding)[0]
                    best_class_indices = np.argmax(predictions)
                    best_class_probability = predictions[best_class_indices]

                    # Get the predicted label
                    predicted_label = self.le.inverse_transform([best_class_indices])[0]

                    # Calculate cosine similarity with all training samples
                    similarities = cosine_similarity(embedding, self.training_data)
                    max_similarity = np.max(similarities)

                    # Thresholds
                    identification_threshold = 0.6  # For recognizing known faces
                    verification_threshold = 0.5  # For confirming identity

                    is_recognized = best_class_probability > identification_threshold
                    is_verified = max_similarity > verification_threshold

                    # Generate a unique face ID
                    face_id = f"{screenshot_id}_{int(time.time() * 1000)}"

                    # Create new filename for the verified face
                    face_filename = f"face_{face_id}.jpg"
                    verified_face_path = os.path.join(self.faces_dir, face_filename)

                    result_temp = {
                        "face_path": verified_face_path,
                        "face_id": face_id,
                        "screenshot_id": screenshot_id,
                        "timestamp": current_time,
                    }

                    result.append(result_temp)

                    # If verified, save the face image and append to JSON
                    if is_verified:
                        # Save the face image
                        cv2.imwrite(
                            verified_face_path,
                            cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR),
                        )
                        # Append to JSON file
                        self._append_to_verified_faces(result_temp)

                except Exception as e:
                    print(f"Error processing {cropped_face_path}: {str(e)}")
                    continue
                finally:
                    os.remove(cropped_face_path)

            return result
        except Exception as outer_e:
            print(f"Outer error: {str(outer_e)}")
            return []

    def _append_to_verified_faces(self, face_data):
        """Append verified face data to JSON file"""
        try:
            # Read existing data
            existing_data = []
            if os.path.exists(self.faces_json_path):
                with open(self.faces_json_path, "r") as f:
                    existing_data = json.load(f)

            # Append new data
            existing_data.append(face_data)

            # Write back to file
            with open(self.faces_json_path, "w") as f:
                json.dump(existing_data, f, indent=2)

        except Exception as e:
            print(f"Error saving verified face data: {str(e)}")
