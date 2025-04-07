import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils

# Constants
EYE_AR_THRESH = 0.25  # Eye aspect ratio threshold for blink detection
EYE_AR_CONSEC_FRAMES = 3  # Minimum consecutive frames for drowsiness flag
HEAD_TILT_THRESH = 15  # Degrees threshold for head tilt detection

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download from dlib.net

# Helper functions
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def get_head_pose(shape, frame):
    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),   # Left eye corner
        (225.0, 170.0, -135.0),    # Right eye corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0)    # Right mouth corner
    ])

    image_points = np.array([
        shape[30],  # Nose tip
        shape[8],    # Chin
        shape[36],   # Left eye corner
        shape[45],   # Right eye corner
        shape[48],   # Left mouth corner
        shape[54]    # Right mouth corner
    ], dtype="double")

    size = frame.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4,1))
    _, rotation_vec, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    rmat, _ = cv2.Rodrigues(rotation_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    return angles

# Initialize counters
blink_counter = 0
drowsy_frames = 0
alarm_on = False

# Start video stream
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    ear = 0  # Default EAR value if no face is detected
    angles = (0, 0, 0)  # Default angles if no face is detected

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        left_eye = shape[36:42]
        right_eye = shape[42:48]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        angles = get_head_pose(shape, frame)
        head_tilt = angles[0]
        head_nod = angles[1]

        if ear < EYE_AR_THRESH:
            drowsy_frames += 1
            if drowsy_frames >= EYE_AR_CONSEC_FRAMES:
                if not alarm_on:
                    alarm_on = True
                    print("ALERT: Drowsiness detected (eyes closed)!")
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            drowsy_frames = 0
            alarm_on = False

        if abs(head_tilt) > HEAD_TILT_THRESH or abs(head_nod) > HEAD_TILT_THRESH:
            cv2.putText(frame, "HEAD TILT DETECTED!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            print(f"Head tilt detected: {head_tilt:.1f}°, nod: {head_nod:.1f}°")

        cv2.drawContours(frame, [left_eye], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye], -1, (0, 255, 0), 1)

    cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Head Tilt: {angles[0]:.1f}°", (300, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Head Nod: {angles[1]:.1f}°", (300, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
