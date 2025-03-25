import os
import time
import shutil
import json
from datetime import datetime
import pyautogui
import threading
import numpy as np
from PIL import Image
import imagehash
import cv2
from .jsonencoder import JSONEncoder
from .face_detector import FaceDetector


class ScreenshotManager:
    def __init__(
        self,
        save_dir="screenshots",
        json_file="screenshot_data.json",
        interval=1,
    ):
        """Initialize the screenshot manager."""
        self.save_dir = save_dir
        self.json_file = json_file
        self.interval = interval

        # Clean up previous runs
        if os.path.exists(self.json_file):
            os.remove(self.json_file)
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)

        # Create directories
        os.makedirs(self.save_dir)
        self.faces_dir = os.path.join(self.save_dir, "faces")
        os.makedirs(self.faces_dir)

        # Initialize face detector
        self.face_detector = FaceDetector()

        # Initialize data structure
        self.data = {
            "screenshots": [],
            "total_count": 0,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cumulative_stats": {
                "total_faces": 0,
            },
            "settings": {
                "interval": self.interval,
                "save_directory": save_dir,
            },
        }
        self._save_json()

        # Store the previous screenshot for comparison
        self.previous_hash = None

        # Thread control
        self.running = False
        self.thread = None

    def _save_json(self):
        """Save the current data to the JSON file."""
        with open(self.json_file, "w") as f:
            json.dump(self.data, f, indent=4, cls=JSONEncoder)

    def _is_same_as_previous(self, screenshot):
        """Compare the current screenshot with the previous one."""
        if self.previous_hash is None:
            return False

        current_hash = imagehash.phash(screenshot)
        hash_difference = current_hash - self.previous_hash
        return hash_difference < 3

    def take_screenshot(self):
        """Take a screenshot and detect faces."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.png"
        filepath = os.path.join(self.save_dir, filename)

        # Take and save screenshot
        screenshot = pyautogui.screenshot()
        is_same = self._is_same_as_previous(screenshot)
        screenshot.save(filepath)
        self.previous_hash = imagehash.phash(screenshot)

        # Detect faces
        image_np = np.array(screenshot)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        face_results = self.face_detector.detect_faces(image_cv)

        # Update cumulative stats
        self.data["cumulative_stats"]["total_faces"] += face_results["total_faces"]

        # Prepare screenshot data
        screenshot_data = {
            "id": self.data["total_count"] + 1,
            "filename": filename,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filepath": filepath,
            "same_as_previous": is_same,
            "face_data": [],
        }

        # Process each face
        for face_info in face_results["faces"]:
            face_data = {
                "face_id": face_info["id"],
                "location": {
                    "top": face_info["location"][0],
                    "right": face_info["location"][1],
                    "bottom": face_info["location"][2],
                    "left": face_info["location"][3],
                },
                "cropped_face_path": None,
            }

            # Save cropped face
            cropped_path = self.face_detector.save_face(
                image_cv,
                face_info["location"],
                self.faces_dir,
                f"face_{face_info['id']}",
            )
            face_data["cropped_face_path"] = cropped_path

            screenshot_data["face_data"].append(face_data)

        # Update data
        self.data["screenshots"].append(screenshot_data)
        self.data["total_count"] += 1
        self._save_json()

        return screenshot_data, is_same

    def _screenshot_loop(self):
        """Main loop for taking screenshots."""
        while self.running:
            screenshot_data, is_same = self.take_screenshot()

            # Print status
            if is_same:
                print(f"Screenshot {screenshot_data['id']} is the same as previous")
            else:
                print(f"Screenshot {screenshot_data['id']} captured:")
                for face in screenshot_data["face_data"]:
                    print(f"  - Detected face {face['face_id']}")

            time.sleep(self.interval)

    def start(self):
        """Start the screenshot capture."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._screenshot_loop)
            self.thread.daemon = True
            self.thread.start()
            print(
                f"Screenshot capture started (Interval: {self.interval}s). "
                f"Saving to {self.save_dir}"
            )

    def stop(self):
        """Stop the screenshot capture."""
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=2)
            print("Screenshot capture stopped.")
            print(f"Results saved to {self.json_file}")
            stats = self.data["cumulative_stats"]
            print(f"Final statistics:")
            print(f"  Total faces detected: {stats['total_faces']}")
