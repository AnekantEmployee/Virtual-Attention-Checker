from deepface import DeepFace
import json
import os
from PIL import Image


class FaceVerification:
    def __init__(self):
        # Load target images
        target_dir = "target_img"  # Directory containing target images
        targets = self.load_target_images(target_dir)
        self.targets = targets
        if not targets:
            raise ValueError("No valid target images found in target directory")

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

    def verifyFace(self):
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

                    best_match = None
                    best_distance = float("inf")

                    for target in self.targets:
                        try:
                            result = DeepFace.verify(
                                img1_path=target["path"],
                                img2_path=cropped_face_path,
                                detector_backend="opencv",
                                enforce_detection=True,
                                distance_metric="cosine",  # Using cosine distance for better comparison
                            )

                            # Track the best match
                            if result["distance"] < best_distance:
                                best_distance = result["distance"]
                                best_match = {
                                    "target_name": target["name"],
                                    "verified": result["verified"],
                                    "distance": result["distance"],
                                    "threshold": result["threshold"],
                                    "similarity": 1
                                    - result[
                                        "distance"
                                    ],  # Convert distance to similarity
                                }

                        except Exception as e:
                            print(
                                f"Error comparing {cropped_face_path} with {target['name']}: {str(e)}"
                            )
                            continue

                    # You can adjust this threshold based on your needs
                    match_threshold = 0.3  # 60% similarity threshold
                    if best_match:
                        is_match = best_match["similarity"] >= match_threshold

                        print(f"\nFace: {cropped_face_path}")
                        print(f"Best match: {best_match['target_name']}")
                        print(f"Similarity: {best_match['similarity']:.2%}")
                        print(f"Threshold: {best_match['threshold']:.2f}")
                        print(f"Verified: {is_match}")
                        print(f"Distance: {best_match['distance']:.4f}")
                    else:
                        print(f"\nFace: {cropped_face_path} - No valid matches found")

        except Exception as outer_e:
            print(f"Outer error: {str(outer_e)}")


face_verification = FaceVerification()
face_verification.verifyFace()
