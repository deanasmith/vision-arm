import cv2
import mediapipe as mp
import numpy as np

# --- Angle Calculation Functions ---
def calculate_3d_angle(a, b, c):
    """Calculates the 3D angle between 3 points (b is the vertex)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    # Handle zero vectors to avoid division by zero
    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0: return 0
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_shoulder_flexion(shoulder, elbow):
    """Calculates front raise angle ignoring the X axis, using a static 'down' vector."""
    # Isolate Y (up/down) and Z (depth)
    sh = np.array([shoulder[1], shoulder[2]]) 
    el = np.array([elbow[1], elbow[2]])
    
    # Vector of the upper arm
    v_arm = el - sh
    # Static vector pointing straight down in Y-Z plane
    v_down = np.array([1.0, 0.0]) 
    
    if np.linalg.norm(v_arm) == 0: return 0
    
    cosine_angle = np.dot(v_arm, v_down) / (np.linalg.norm(v_arm) * np.linalg.norm(v_down))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# --- MediaPipe Setup ---
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

UPPER_ARM_CONNECTIONS = [(11, 12), (11, 13), (12, 14)]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    h, w, _ = frame.shape 
    results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        pose_lms = results.pose_landmarks.landmark
        
        # --- 1. Draw Skeleton (Shoulders & Upper Arms) ---
        for connection in UPPER_ARM_CONNECTIONS:
            pt1, pt2 = pose_lms[connection[0]], pose_lms[connection[1]]
            if pt1.visibility > 0.5 and pt2.visibility > 0.5:
                x1, y1 = int(pt1.x * w), int(pt1.y * h)
                x2, y2 = int(pt2.x * w), int(pt2.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.circle(frame, (x1, y1), 5, (0, 0, 255), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 5, (0, 0, 255), cv2.FILLED)

        # --- 2. Left Arm Logic (Forearm & Math) ---
        if pose_lms[11].visibility > 0.5 and pose_lms[13].visibility > 0.5:
            elbow_x, elbow_y = int(pose_lms[13].x * w), int(pose_lms[13].y * h)
            
            # 1. ALWAYS calculate the Shoulder Angle (only needs Shoulder + Elbow)
            shoulder_3d = [pose_lms[11].x, pose_lms[11].y, pose_lms[11].z]
            elbow_3d = [pose_lms[13].x, pose_lms[13].y, pose_lms[13].z]
            shoulder_angle = calculate_shoulder_flexion(shoulder_3d, elbow_3d)
            
            shoulder_px = (int(pose_lms[11].x * w), int(pose_lms[11].y * h))
            cv2.putText(frame, f"Shoulder: {int(shoulder_angle)}", (shoulder_px[0] + 15, shoulder_px[1] - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Determine Wrist Position
            if results.left_hand_landmarks:
                wrist_x = int(results.left_hand_landmarks.landmark[0].x * w)
                wrist_y = int(results.left_hand_landmarks.landmark[0].y * h)
                wrist_z = pose_lms[15].z 
            elif pose_lms[15].visibility > 0.5:
                wrist_x = int(pose_lms[15].x * w)
                wrist_y = int(pose_lms[15].y * h)
                wrist_z = pose_lms[15].z
            else:
                wrist_x, wrist_y, wrist_z = None, None, None

            # 2. Only calculate the Elbow Angle if the Wrist is visible
            if wrist_x is not None:
                cv2.line(frame, (elbow_x, elbow_y), (wrist_x, wrist_y), (0, 255, 0), 3)
                cv2.circle(frame, (wrist_x, wrist_y), 5, (0, 0, 255), cv2.FILLED)
                
                wrist_3d = [pose_lms[15].x, pose_lms[15].y, wrist_z] 
                elbow_angle = calculate_3d_angle(shoulder_3d, elbow_3d, wrist_3d)

                cv2.putText(frame, f"Elbow: {int(elbow_angle)}", (elbow_x + 15, elbow_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # --- 3. Right Arm Logic (Forearm Only) ---
        if pose_lms[14].visibility > 0.5:
            elbow_x, elbow_y = int(pose_lms[14].x * w), int(pose_lms[14].y * h)
            if results.right_hand_landmarks:
                wrist_x = int(results.right_hand_landmarks.landmark[0].x * w)
                wrist_y = int(results.right_hand_landmarks.landmark[0].y * h)
            elif pose_lms[16].visibility > 0.5:
                wrist_x = int(pose_lms[16].x * w)
                wrist_y = int(pose_lms[16].y * h)
            else:
                wrist_x, wrist_y = None, None

            if wrist_x is not None:
                cv2.line(frame, (elbow_x, elbow_y), (wrist_x, wrist_y), (0, 255, 0), 3)
                cv2.circle(frame, (wrist_x, wrist_y), 5, (0, 0, 255), cv2.FILLED)

    # --- 4. Draw Hands ---
    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    cv2.imshow('Kinematic Arm Tracking', frame)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()