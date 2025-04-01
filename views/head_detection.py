import cv2
import numpy as np
import mediapipe as mp


class HeadPoseDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Landmark indices for head pose estimation
        self.HEAD_POSE_LANDMARKS = [1, 33, 61, 199, 263, 291]

    def calculate_head_pose(self, landmarks, image_shape):
        """Estimate head pose angles (pitch and yaw)"""
        model_points = np.array(
            [
                (0.0, 0.0, 0.0),  # Nose tip
                (0.0, -330.0, -65.0),  # Chin
                (-225.0, 170.0, -135.0),  # Left eye left corner
                (225.0, 170.0, -135.0),  # Right eye right corner
                (-150.0, -150.0, -125.0),  # Left mouth corner
                (150.0, -150.0, -125.0),  # Right mouth corner
            ]
        )

        image_points = np.array(
            [
                (
                    landmarks.landmark[1].x * image_shape[1],
                    landmarks.landmark[1].y * image_shape[0],
                ),
                (
                    landmarks.landmark[199].x * image_shape[1],
                    landmarks.landmark[199].y * image_shape[0],
                ),
                (
                    landmarks.landmark[33].x * image_shape[1],
                    landmarks.landmark[33].y * image_shape[0],
                ),
                (
                    landmarks.landmark[263].x * image_shape[1],
                    landmarks.landmark[263].y * image_shape[0],
                ),
                (
                    landmarks.landmark[61].x * image_shape[1],
                    landmarks.landmark[61].y * image_shape[0],
                ),
                (
                    landmarks.landmark[291].x * image_shape[1],
                    landmarks.landmark[291].y * image_shape[0],
                ),
            ],
            dtype="double",
        )

        # Camera matrix
        focal_length = image_shape[1]
        center = (image_shape[1] / 2, image_shape[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )

        dist_coeffs = np.zeros((4, 1))
        _, rotation_vector, _ = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        # Convert rotation vector to matrix
        rmat, _ = cv2.Rodrigues(rotation_vector)

        # Calculate angles
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        return {
            "pitch": angles[0],  # Vertical head movement (nodding)
            "yaw": angles[1],  # Horizontal head movement (shaking)
        }

    def process_image(self, image_pil):
        """Process single image and return head pose data"""
        image = np.array(image_pil)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert PIL RGB to OpenCV BGR

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Calculate head pose
                head_angles = self.calculate_head_pose(face_landmarks, image.shape)
                return {
                    "head_pitch": round(head_angles["pitch"], 2),
                    "head_yaw": round(head_angles["yaw"], 2),
                    "is_verified": True,
                }

        return {"head_pitch": 0, "head_yaw": 0, "is_verified": False}
