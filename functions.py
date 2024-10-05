import math
import cv2
import mediapipe as mp
import numpy as np
import time  # For handling timers

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

# Define thresholds for detecting attention states
PITCH_FORWARD_THRESH = 15  # Allow slight pitch movement for "paying attention"
YAW_LEFT_THRESH = -40
YAW_RIGHT_THRESH = 40
PITCH_DOWN_THRESH = -30  # More lenient pitch down for note-taking

# Initialize variables to track states and timers
current_state = "paying attention"
thinking_start_time = None
notes_start_time = None
DISTRACTED_TIME_THINKING = 5  # seconds
DISTRACTED_TIME_NOTES = 5  # seconds


def rotation_matrix_to_angles(rotation_matrix):
    """
    Calculate Euler angles from rotation matrix.
    :param rotation_matrix: A 3*3 matrix
    :return: Angles in degrees for each axis (pitch, yaw, roll)
    """
    x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 +
                                                     rotation_matrix[1, 0] ** 2))
    z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return np.array([x, y, z]) * 180. / math.pi


def update_state(pitch, yaw):
    global current_state, thinking_start_time, notes_start_time

    # Broaden thresholds for "paying attention"
    if -PITCH_FORWARD_THRESH <= pitch <= PITCH_FORWARD_THRESH and abs(yaw) <= 15:
        current_state = "paying attention"
        thinking_start_time = None
        notes_start_time = None

    # Check for "taking notes" before "thinking"
    elif pitch < PITCH_DOWN_THRESH:
        if current_state != "taking notes" and current_state != "distracted2" and current_state != "distracted":
            current_state = "taking notes"
            if notes_start_time is None:  # Only set the start time once
                notes_start_time = time.time()
        elif notes_start_time and time.time() - notes_start_time > DISTRACTED_TIME_NOTES:
            current_state = "distracted2"
            notes_start_time = None

    # Consider "thinking" based on yaw and pitch
    elif yaw < YAW_LEFT_THRESH or yaw > YAW_RIGHT_THRESH or pitch > PITCH_FORWARD_THRESH:
        if current_state != "thinking" and current_state != "distracted2" and current_state != "distracted":
            current_state = "thinking"
            if thinking_start_time is None:  # Only set the start time once
                thinking_start_time = time.time()
        elif thinking_start_time and time.time() - thinking_start_time > DISTRACTED_TIME_THINKING:
            current_state = "distracted"
            thinking_start_time = None


while cap.isOpened():
    success, image = cap.read()

    # Convert the color space from BGR to RGB and get Mediapipe results
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Convert the color space from RGB to BGR for display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    face_coordination_in_real_world = np.array([
        [285, 528, 200],
        [285, 371, 152],
        [197, 574, 128],
        [173, 425, 108],
        [360, 574, 128],
        [391, 425, 108]
    ], dtype=np.float64)

    h, w, _ = image.shape
    face_coordination_in_image = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [1, 9, 57, 130, 287, 359]:
                    x, y = int(lm.x * w), int(lm.y * h)
                    face_coordination_in_image.append([x, y])

            face_coordination_in_image = np.array(face_coordination_in_image,
                                                  dtype=np.float64)

            # The camera matrix
            focal_length = 1 * w
            cam_matrix = np.array([[focal_length, 0, w / 2],
                                   [0, focal_length, h / 2],
                                   [0, 0, 1]])

            # The Distance Matrix
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Use solvePnP function to get rotation vector
            success, rotation_vec, transition_vec = cv2.solvePnP(
                face_coordination_in_real_world, face_coordination_in_image,
                cam_matrix, dist_matrix)

            # Use Rodrigues function to convert rotation vector to matrix
            rotation_matrix, jacobian = cv2.Rodrigues(rotation_vec)

            result = rotation_matrix_to_angles(rotation_matrix)
            pitch, yaw, roll = result

            # Update the attention state based on the pitch and yaw values
            update_state(pitch, yaw)

            for i, info in enumerate(zip(('pitch', 'yaw', 'roll'), result)):
                k, v = info
                text = f'{k}: {int(v)}'
                cv2.putText(image, text, (20, i * 30 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 0, 200), 2)

    # Display the current state on the image
    cv2.putText(image, f'State: {current_state}', (20, h - 30),  # Show at bottom left
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    cv2.imshow('Head Pose Angles', image)

    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
