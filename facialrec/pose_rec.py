import cv2
import numpy
import mediapipe as mp
mp_pose = mp.solutions.pose

class PostureRec:
    def detect_posture(self, image):
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
            image = cv2.imdecode(numpy.fromstring(image.read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.pose_landmarks:
                return "ERROR"
            # Shoulders
            right_shoulder = results.pose_landmarks.landmark[12]
            left_shoulder = results.pose_landmarks.landmark[11]
            right_hip = results.pose_landmarks.landmark[24]
            left_hip = results.pose_landmarks.landmark[23]
            # Coding the logic
            if (right_shoulder.x > right_hip.x):
                return "TILTED LEFT"
            if (left_shoulder.x < left_hip.x):
                return "TILTED RIGHT"
            return "GOOD"