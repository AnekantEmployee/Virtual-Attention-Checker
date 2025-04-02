import os
import cv2
import time
import mtcnn
import torch
import joblib
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from datetime import datetime
from .eye_detection import EyeDetector
from .yawn_detection import YawnDetector
from .head_detection import HeadPoseDetector
from facenet_pytorch import InceptionResnetV1
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity


class FaceVerification:
    def __init__(self):
        # Initialize face detector with min_face_size parameter
        self.face_detector = mtcnn.MTCNN()

        # Initialize FaceNet for face embeddings
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval()

        # Headpose detector
        self.head_pose_detector = HeadPoseDetector()
        self.yawn_detector = YawnDetector()
        self.eye_detection = EyeDetector()

        # Create directories for verified faces
        self.faces_dir = "faces"
        self.model_dir = "models/verification"  # Directory to store trained models

        # Create directories if they don't exist
        os.makedirs(self.faces_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        # Load target images and train model
        target_dir = "target_img"
        self.targets = self.load_target_images(target_dir)
        if not self.targets:
            raise ValueError("No valid target images found in target directory")

        # Check if we can load a cached model
        if not self.load_cached_model():
            # No cached model available or it's outdated, train a new one
            self.train_model()
            self.cache_model()

    def get_target_images_hash(self):
        """Generate a hash of the target images to detect changes"""
        import hashlib

        hash_obj = hashlib.sha256()
        for target in sorted(self.targets, key=lambda x: x["name"]):
            with open(target["path"], "rb") as f:
                while chunk := f.read(8192):
                    hash_obj.update(chunk)
        return hash_obj.hexdigest()

    def load_cached_model(self):
        """Try to load a cached model if available and still valid"""
        try:
            # Check if we have a cached model
            model_path = os.path.join(self.model_dir, "face_verification_model.joblib")
            hash_path = os.path.join(self.model_dir, "target_images_hash.txt")

            if not os.path.exists(model_path) or not os.path.exists(hash_path):
                return False

            # Check if the current target images match the cached ones
            with open(hash_path, "r") as f:
                cached_hash = f.read().strip()

            current_hash = self.get_target_images_hash()
            if cached_hash != current_hash:
                return False

            # Load the cached model
            cached_data = joblib.load(model_path)
            self.classifier = cached_data["classifier"]
            self.le = cached_data["label_encoder"]
            self.training_data = cached_data["training_data"]
            self.training_labels = cached_data["training_labels"]

            print("Loaded cached face verification model")
            return True

        except Exception as e:
            print(f"Error loading cached model: {str(e)}")
            return False

    def cache_model(self):
        """Cache the current trained model"""
        try:
            model_path = os.path.join(self.model_dir, "face_verification_model.joblib")
            hash_path = os.path.join(self.model_dir, "target_images_hash.txt")

            # Save model data
            model_data = {
                "classifier": self.classifier,
                "label_encoder": self.le,
                "training_data": self.training_data,
                "training_labels": self.training_labels,
            }

            joblib.dump(model_data, model_path)

            # Save current target images hash
            current_hash = self.get_target_images_hash()
            with open(hash_path, "w") as f:
                f.write(current_hash)

            print("Successfully cached face verification model")

        except Exception as e:
            print(f"Error caching model: {str(e)}")

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

    def verifyFace(self, face_data_pil):
        """Verify faces from PIL Image using our improved model"""
        try:
            current_time = datetime.now().isoformat()
            face_id = f"{int(time.time() * 1000)}"  # Generate a unique face ID

            try:
                # Convert PIL Image to numpy array
                img_rgb = np.array(face_data_pil)

                # Handle different image modes
                if img_rgb.ndim == 2:  # Convert grayscale to RGB
                    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
                elif img_rgb.shape[2] == 4:  # Convert RGBA to RGB
                    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)

                # Detect face with MTCNN
                face_results = self.face_detector.detect_faces(img_rgb)
                if len(face_results) == 0:
                    print("No face detected in the provided image")
                    return {
                        "is_verified": False,
                        "timestamp": datetime.now().isoformat(),
                    }

                # Take the best face
                best_face = max(face_results, key=lambda x: x["confidence"])
                x, y, w, h = best_face["box"]

                # Ensure coordinates are within bounds
                height, width = img_rgb.shape[:2]
                x, y = max(0, x), max(0, y)
                w, h = min(w, width - x), min(h, height - y)

                if w <= 0 or h <= 0:
                    print("Invalid face dimensions in the provided image")
                    return {
                        "is_verified": False,
                        "timestamp": datetime.now().isoformat(),
                    }

                face_img = img_rgb[y : y + h, x : x + w]

                # Get embedding
                embedding = self.get_face_embedding(face_img)
                if embedding is None:
                    print("Could not generate embedding for the face")
                    return {
                        "is_verified": False,
                        "timestamp": datetime.now().isoformat(),
                    }

                embedding = embedding.reshape(1, -1)

                # Calculate cosine similarity with all training samples
                similarities = cosine_similarity(embedding, self.training_data)
                max_similarity = np.max(similarities)

                # Thresholds
                verification_threshold = 0.5  # For confirming identity
                is_verified = max_similarity > verification_threshold

                # Save the verified face
                face_filename = f"face_{face_id}.jpg"
                verified_face_path = os.path.join(self.faces_dir, face_filename)

                # If verified, append to JSON
                if is_verified:
                    verified_face_pil = Image.fromarray(face_img)
                    verified_face_pil.save(verified_face_path)
                    head_pose_results = self.head_pose_detector.process_image(face_img)
                    eye_results = self.eye_detection.get_eye_state(face_img)
                    yawn_results = self.yawn_detector.process_image(face_img)

                    result_temp = {
                        "is_verified": True,
                        "timestamp": current_time,
                        "face_path": verified_face_path,
                        "eye_state": eye_results["eye_state"],
                        "head_yaw": head_pose_results["head_yaw"],
                        "head_pitch": head_pose_results["head_pitch"],
                        "yawn_results": (
                            -1
                            if not yawn_results["is_verified"]
                            else yawn_results["yawn_state"]
                        ),
                        "presenting_state": 0,
                    }
                    # print(result_temp)
                    print(f"Face Processed Succesfully for {verified_face_path}")
                    return result_temp
                else:
                    return {
                        "is_verified": False,
                        "timestamp": datetime.now().isoformat(),
                    }

            except Exception as e:
                print(f"Error processing face image: {str(e)}")
                return {
                    "is_verified": False,
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as outer_e:
            print(f"Outer error: {str(outer_e)}")
            return {
                "is_verified": False,
                "timestamp": datetime.now().isoformat(),
            }
