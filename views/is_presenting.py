import cv2
import numpy as np


class PresentingChecker:
    def __init__(
        self,
        border_width=2,
        red_threshold=200,
        red_percentage_threshold=0.8,
    ):
        """
        Initializes the PresentingChecker object.

        Args:
            border_width: Width of the border to check (1-2 pixels).
            red_threshold: Minimum value for red channel (0-255).
            red_percentage_threshold: Minimum percentage of red pixels to consider a border as red.
            screenshots_dir: Directory to save screenshots.
        """
        self.border_width = border_width
        self.red_threshold = red_threshold
        self.red_percentage_threshold = red_percentage_threshold

    def has_red_border(self, image_cv2):
        """
        Check if the image has a red border around its edges.

        Args:
            image_cv2: Input image in BGR format (OpenCV format)

        Returns:
            bool: True if red border exists, False otherwise
        """
        if image_cv2 is None or image_cv2.size == 0:
            return False

        height, width = image_cv2.shape[:2]

        # Define the border regions to check
        top_border = image_cv2[0 : self.border_width, :]
        bottom_border = image_cv2[height - self.border_width : height, :]
        left_border = image_cv2[:, 0 : self.border_width]
        right_border = image_cv2[:, width - self.border_width : width]

        # Check each border for redness
        borders = [top_border, bottom_border, left_border, right_border]

        for border in borders:
            # Split into channels (BGR format)
            blue, green, red = cv2.split(border)

            # Check if red is dominant (red > threshold and red > blue/green by significant margin)
            red_dominant = (
                (red > self.red_threshold) & (red > blue * 1.5) & (red > green * 1.5)
            )

            # At least 80% of the border pixels should be red
            if np.mean(red_dominant) < self.red_percentage_threshold:
                return False

        return True
