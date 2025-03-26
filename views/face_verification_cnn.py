import os
import json
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle
from PIL import Image

class FaceVerification:
    def __init__(self):
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load target images and train KNN model
        target_dir = "target_img"
        self.targets = self.load_target_images(target_dir)
        if not self.targets:
            raise ValueError("No valid target images found in target directory")
        
        # Train KNN model
        self.train_knn_model()

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
                    targets.append({
                        "name": os.path.splitext(filename)[0],
                        "path": path
                    })
                else:
                    print(f"Skipping target {filename}: {msg}")
        return targets

    def train_knn_model(self):
        """Train KNN model with target images"""
        face_data = []
        labels = []
        
        for target in self.targets:
            try:
                # Read and convert to grayscale
                img = cv2.imread(target["path"])
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Detect face
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) == 0:
                    print(f"No face detected in target image: {target['path']}")
                    continue
                
                # Take the first face found
                (x, y, w, h) = faces[0]
                face_img = gray[y:y+h, x:x+w]
                
                # Resize and flatten
                resized_face = cv2.resize(face_img, (50, 50)).flatten()
                face_data.append(resized_face)
                labels.append(target["name"])
                
            except Exception as e:
                print(f"Error processing target image {target['path']}: {str(e)}")
                continue
        
        if len(face_data) == 0:
            raise ValueError("No valid face data found in target images")
        
        # Train KNN classifier
        self.knn = KNeighborsClassifier(n_neighbors=1)
        self.knn.fit(face_data, labels)
        
        # Store the face data and labels for distance calculation
        self.training_data = np.array(face_data)
        self.training_labels = np.array(labels)

    def verifyFace(self):
        """Verify faces in screenshots using KNN"""
        try:
            # Load screenshot data
            with open("screenshot_data.json") as f:
                data = json.load(f)["screenshots"]

            for screenshot in data:
                if not screenshot["face_data"]:
                    continue

                for face in screenshot["face_data"]:
                    cropped_face_path = face["cropped_face_path"]

                    # Check if face image is valid
                    valid, msg = self.is_valid_image(cropped_face_path)
                    if not valid:
                        print(f"Skipping {cropped_face_path}: {msg}")
                        continue

                    try:
                        # Process the face image
                        img = cv2.imread(cropped_face_path)
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                        # Resize and flatten
                        resized_face = cv2.resize(gray, (50, 50)).flatten().reshape(1, -1)
                        
                        # Predict using KNN
                        predicted_label = self.knn.predict(resized_face)[0]
                        
                        # Calculate distances to all training samples
                        distances, indices = self.knn.kneighbors(resized_face)
                        
                        # Get the closest distance
                        min_distance = distances[0][0]
                        
                        # Calculate similarity (inverse of distance)
                        # Normalize distance to [0,1] range (you may need to adjust this)
                        max_possible_distance = np.sqrt(50*50*255**2)  # Max possible Euclidean distance
                        similarity = 1 - (min_distance / max_possible_distance)
                        
                        # You can adjust this threshold based on your needs
                        match_threshold = 0.7  # 60% similarity threshold
                        is_match = similarity >= match_threshold
                        
                        print(f"\nFace: {cropped_face_path}")
                        print(f"Best match: {predicted_label}")
                        print(f"Similarity: {similarity:.2%}")
                        print(f"Distance: {min_distance:.4f}")
                        print(f"Verified: {is_match}")

                    except Exception as e:
                        print(f"Error processing {cropped_face_path}: {str(e)}")
                        continue

        except Exception as outer_e:
            print(f"Outer error: {str(outer_e)}")


face_verification = FaceVerification()
face_verification.verifyFace()