import os
import cv2
import time
import threading
import pyautogui
import numpy as np
from PIL import Image
import pandas as pd
from .face_detector import FaceDetector


class ScreenshotManager:
    def __init__(
        self,
        interval=1,
        mode="testing",  # "testing" or "training"
        # training_video="D:\DS Project\Face Recoginition Meeting\meeting_recoding.avi",
        training_video="WIN_20250329_00_32_17_Pro.mp4",
    ):
        """Initialize the screenshot manager."""
        self.interval = interval
        self.mode = mode
        self.training_video = training_video
        self.video_capture = None
        self.video_fps = None
        self.frame_interval = None

        # Initialize face detector
        self.face_detector = FaceDetector()
        self.final_output = []

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
        results = self.face_detector.detect_faces(image_cv)
        if self.mode == 'testing' and results['is_verified']:
            self.final_output.append(results)        
        return results

    def _process_video_frames(self):
        """Process video frames at specified intervals in training mode."""
        frame_count = 0
        while self.running:
            # Set the frame position
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

            # Process frame
            screenshot_data = self.take_screenshot()

            try:
                print("Checking", screenshot_data)
                if screenshot_data is None:  # End of video
                    print("Finished processing all frames from training video.")
                    self.stop()  # stop the loop, but do not join the thread.
                    break
                else:
                    if screenshot_data.get("is_verified"):
                        self.final_output.append(screenshot_data)
            except Exception as e:
                print(f"Error: {e}")

            # Move to next frame at specified interval
            frame_count += self.frame_interval
            if frame_count >= self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT):
                break

    def save_results(self):
        path = "final_output.csv"
        if os.path.exists(path):
            os.remove(path)

        df = pd.DataFrame(self.final_output)
        df.to_csv(path)

    def _screenshot_loop(self):
        """Main loop for taking screenshots."""
        if self.mode == "training":
            self._process_video_frames()
        else:  # testing mode
            while self.running:
                self.take_screenshot()
                # time.sleep(self.interval)

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
            )

    def stop(self):
        """Stop the screenshot capture."""
        self.save_results()
        if self.running:
            self.running = False
            if self.video_capture is not None:
                self.video_capture.release()
            print("Screenshot capture stopped.")
