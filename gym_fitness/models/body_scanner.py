import cv2

try:

    import mediapipe as mp

except ImportError:

    mp = None


class BodyScanner:

    def __init__(self):

        self.enabled = (
            mp is not None
        )


        self.pose = None


        if self.enabled:

            self.pose = mp.solutions.pose.Pose(

                static_image_mode=False,

                model_complexity=1,

                min_detection_confidence=0.5,

                min_tracking_confidence=0.5

            )


    def process_frame(
        self,
        frame
    ):

        if not self.enabled:

            return frame, []


        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        results = self.pose.process(
            rgb_frame
        )


        landmarks = []


        if results.pose_landmarks:

            for landmark in results.pose_landmarks.landmark:

                landmarks.append({

                    "x": round(
                        landmark.x,
                        4
                    ),

                    "y": round(
                        landmark.y,
                        4
                    ),

                    "z": round(
                        landmark.z,
                        4
                    ),

                    "visibility": round(
                        landmark.visibility,
                        4
                    )
                })


        return frame, landmarks