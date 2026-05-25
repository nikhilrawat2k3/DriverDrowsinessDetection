import cv2
import time
import math
import mediapipe as mp
from mediapipe.python.solutions import face_mesh

import numpy as np

# Optional sound on Windows
try:
    import winsound

    def play_alarm():
        winsound.Beep(2500, 300)
except ImportError:
    def play_alarm():
        print("\a", end="", flush=True)

EYE_AR_THRESH = 0.21
EYE_AR_CONSEC_FRAMES = 20

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

mp_face_mesh = face_mesh
mp_drawing = mp.solutions.drawing_utils


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def eye_aspect_ratio(eye_points):
    # eye_points = [p1, p2, p3, p4, p5, p6]
    # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    A = euclidean_distance(eye_points[1], eye_points[5])
    B = euclidean_distance(eye_points[2], eye_points[4])
    C = euclidean_distance(eye_points[0], eye_points[3])

    if C == 0:
        return 0.0

    ear = (A + B) / (2.0 * C)
    return ear


def landmark_to_point(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)


def get_eye_points(landmarks, eye_indices, width, height):
    points = []
    for idx in eye_indices:
        points.append(landmark_to_point(landmarks[idx], width, height))
    return points


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    counter = 0
    drowsy = False
    start_time = time.time()

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb_frame)

            if result.multi_face_landmarks:
                face_landmarks = result.multi_face_landmarks[0].landmark

                left_eye_points = get_eye_points(face_landmarks, LEFT_EYE, w, h)
                right_eye_points = get_eye_points(face_landmarks, RIGHT_EYE, w, h)

                left_ear = eye_aspect_ratio(left_eye_points)
                right_ear = eye_aspect_ratio(right_eye_points)
                ear = (left_ear + right_ear) / 2.0

                # Draw eye landmarks
                for pt in left_eye_points + right_eye_points:
                    cv2.circle(frame, pt, 2, (0, 255, 0), -1)

                if ear < EYE_AR_THRESH:
                    counter += 1
                    if counter >= EYE_AR_CONSEC_FRAMES:
                        drowsy = True
                        play_alarm()
                        cv2.putText(
                            frame,
                            "DROWSINESS ALERT!",
                            (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0, 0, 255),
                            3
                        )
                else:
                    counter = 0
                    drowsy = False

                # Display EAR
                cv2.putText(
                    frame,
                    f"EAR: {ear:.2f}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2
                )

                # Show status
                status_text = "DROWSY" if drowsy else "ALERT"
                status_color = (0, 0, 255) if drowsy else (0, 255, 0)
                cv2.putText(
                    frame,
                    f"STATUS: {status_text}",
                    (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    status_color,
                    2
                )
            else:
                cv2.putText(
                    frame,
                    "No face detected",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2
                )

            fps = 1.0 / max(time.time() - start_time, 0.001)
            start_time = time.time()

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (50, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 0),
                2
            )

            cv2.imshow("Driver Drowsiness Detector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
