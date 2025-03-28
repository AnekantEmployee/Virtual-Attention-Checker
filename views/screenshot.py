import os
import cv2
import json
import time
import shutil
import threading
import pyautogui
import numpy as np
from PIL import Image
from datetime import datetime
from .jsonencoder import JSONEncoder
from .face_detector import FaceDetector
from .face_verification import FaceVerification


class ScreenshotManager:
    def __init__(
        self,
        save_dir="screenshots",
        json_file="screenshot_data.json",
        interval=1,
        mode="training",  # "testing" or "training"
        training_video="D:\DS Project\Face Recoginition Meeting\meeting_recoding.avi",
    ):
        """Initialize the screenshot manager."""
        self.save_dir = save_dir
        self.json_file = json_file
        self.interval = interval
        self.mode = mode
        self.training_video = training_video
        self.video_capture = None
        self.video_fps = None
        self.frame_interval = None

        # Clean up previous runs with error handling
        try:
            if os.path.exists(self.json_file):
                os.remove(self.json_file)
        except PermissionError:
            print(f"Warning: Could not delete {self.json_file} - file in use")

        # More robust directory cleanup
        try:
            if os.path.exists(self.save_dir):
                # Remove all contents first
                for filename in os.listdir(self.save_dir):
                    file_path = os.path.join(self.save_dir, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"Failed to delete {file_path}. Reason: {e}")
                # Then remove the directory itself
                shutil.rmtree(self.save_dir)
        except Exception as e:
            print(f"Warning: Could not delete {self.save_dir} - {str(e)}")

        # Create directories with exist_ok=True to prevent errors if directory exists
        os.makedirs(self.save_dir, exist_ok=True)
        self.faces_dir = os.path.join(self.save_dir, "faces")
        os.makedirs(self.faces_dir, exist_ok=True)

        # Initialize face detector
        self.face_detector = FaceDetector()
        self.face_verification = FaceVerification()

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
                "mode": mode,
            },
        }
        self._save_json()

        # Store the previous screenshot for comparison
        self.previous_hash = None

        # Thread control
        self.running = False
        self.thread = None

        # Initialize video capture if in training mode
        if self.mode == "training":
            if not os.path.exists(self.training_video):
                raise FileNotFoundError(
                    f"Training video file not found: {self.training_video}"
                )
            self.video_capture = cv2.VideoCapture(self.training_video)
            if not self.video_capture.isOpened():
                raise ValueError(f"Could not open video file: {self.training_video}")

            # Get video properties
            self.video_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            self.frame_interval = int(self.video_fps * self.interval)

    def _save_json(self):
        """Save the current data to the JSON file."""
        with open(self.json_file, "w") as f:
            json.dump(self.data, f, indent=4, cls=JSONEncoder)

    def _get_screenshot(self):
        """Get screenshot based on current mode."""
        if self.mode == "testing":
            # Take live screenshot
            screenshot = pyautogui.screenshot()
            image_np = np.array(screenshot)
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            return screenshot, image_cv
        else:  # training mode
            # Read frame from video
            ret, frame = self.video_capture.read()
            if not ret:
                return None, None

            # Convert OpenCV BGR to RGB for PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            screenshot = Image.fromarray(frame_rgb)
            return screenshot, frame

    def take_screenshot(self):
        """Take a screenshot and detect faces."""

        # Get screenshot based on mode
        screenshot, image_cv = self._get_screenshot()
        if screenshot is None:  # End of video in training mode
            return None, True

        # Detect faces
        face_results = self.face_detector.detect_faces(image_cv)

        # Update cumulative stats
        self.data["cumulative_stats"]["total_faces"] += face_results["total_faces"]

        # Prepare screenshot data
        screenshot_data = {
            "id": self.data["total_count"] + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "face_data": [],
        }
        
        # Process each face
        for face_info in face_results["faces"]:
            face_data = {
                "face_id": face_info["id"],
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

            # Appending final data
            screenshot_data["face_data"].append(face_data)

        # Update data
        screenshot_data["verification_data"] = self.face_verification.verifyFace(
            screenshot_data
        )
        self.data["screenshots"].append(screenshot_data)
        self.data["total_count"] += 1

        self._save_json()

        return screenshot_data

    def _process_video_frames(self):
        """Process video frames at specified intervals in training mode."""
        frame_count = 0
        while self.running:
            # Set the frame position
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

            # Process frame
            screenshot_data = self.take_screenshot()

            if screenshot_data is None:  # End of video
                print("Finished processing all frames from training video.")
                self.stop()
                break

            # Print status
            print(f"Processed frame {screenshot_data['id']}:")
            for face in screenshot_data["face_data"]:
                print(f"  - Detected face {face['face_id']}")

            # Move to next frame at specified interval
            frame_count += self.frame_interval
            if frame_count >= self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT):
                break

    def _screenshot_loop(self):
        """Main loop for taking screenshots."""
        if self.mode == "training":
            self._process_video_frames()
        else:  # testing mode
            while self.running:
                screenshot_data, is_same = self.take_screenshot()

                # Print status
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
            mode_str = "live screen" if self.mode == "testing" else "training video"
            print(
                f"Screenshot capture started from {mode_str} (Interval: {self.interval}s). "
                f"Saving to {self.save_dir}"
            )

    def stop(self):
        """Stop the screenshot capture."""
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=2)
            if self.video_capture is not None:
                self.video_capture.release()
            print("Screenshot capture stopped.")
            print(f"Results saved to {self.json_file}")
            stats = self.data["cumulative_stats"]
            print(f"Final statistics:")
            print(f"  Total faces detected: {stats['total_faces']}")
