import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from screeninfo import get_monitors
#smoothness is basically frame to frame displacement consistancy..
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

FONT = cv2.FONT_HERSHEY_SIMPLEX

monitor = get_monitors()[0]
screen_w, screen_h = monitor.width, monitor.height

WINDOW_NAME = "Stroke Rehab – Full System"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, screen_w, screen_h)

DETECTION_WINDOW = int(7 * fps)
ASYMMETRY_THRESHOLD = 0.80
LOCK_CONFIDENCE = 0.35

left_arm_motion = right_arm_motion = 0
left_leg_motion = right_leg_motion = 0

prev = {}
frame_count = 0
affected_side = "Detecting..."
locked = False
confidence = 0.0

WINDOW = int(1.5 * fps)
l_wr_hist = deque(maxlen=WINDOW)
r_wr_hist = deque(maxlen=WINDOW)
l_an_hist = deque(maxlen=WINDOW)
r_an_hist = deque(maxlen=WINDOW)

def smoothness_score(history):
    if len(history) < 5:
        return 0.0
    diffs = [abs(history[i] - history[i-1]) for i in range(1, len(history))]
    mean = np.mean(diffs)
    std = np.std(diffs)
    if mean == 0:
        return 0.0
    return max(0.0, min(1 - (std / mean), 1.0))

def draw_right_text(frame, text, y, scale, color, thick=2, margin=10):
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, thick)
    x = frame.shape[1] - tw - margin
    cv2.putText(frame, text, (x, y), FONT, scale, color, thick)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        def P(i): return (int(lm[i].x * w), int(lm[i].y * h))

        l_ear, r_ear = P(mp_pose.PoseLandmark.LEFT_EAR), P(mp_pose.PoseLandmark.RIGHT_EAR)
        l_sh, r_sh = P(mp_pose.PoseLandmark.LEFT_SHOULDER), P(mp_pose.PoseLandmark.RIGHT_SHOULDER)
        l_el, r_el = P(mp_pose.PoseLandmark.LEFT_ELBOW), P(mp_pose.PoseLandmark.RIGHT_ELBOW)
        l_wr, r_wr = P(mp_pose.PoseLandmark.LEFT_WRIST), P(mp_pose.PoseLandmark.RIGHT_WRIST)
        l_hip, r_hip = P(mp_pose.PoseLandmark.LEFT_HIP), P(mp_pose.PoseLandmark.RIGHT_HIP)
        l_kn, r_kn = P(mp_pose.PoseLandmark.LEFT_KNEE), P(mp_pose.PoseLandmark.RIGHT_KNEE)
        l_an, r_an = P(mp_pose.PoseLandmark.LEFT_ANKLE), P(mp_pose.PoseLandmark.RIGHT_ANKLE)

        if "l_wr" in prev and not locked:
            left_arm_motion += abs(l_wr[1] - prev["l_wr"][1])
            right_arm_motion += abs(r_wr[1] - prev["r_wr"][1])
            left_leg_motion += abs(l_an[1] - prev["l_an"][1])
            right_leg_motion += abs(r_an[1] - prev["r_an"][1])

        prev["l_wr"], prev["r_wr"] = l_wr, r_wr
        prev["l_an"], prev["r_an"] = l_an, r_an

        if not locked:
            frame_count += 1

        l_wr_hist.append(l_wr[1])
        r_wr_hist.append(r_wr[1])
        l_an_hist.append(l_an[1])
        r_an_hist.append(r_an[1])

        l_arm_smooth = smoothness_score(l_wr_hist)
        r_arm_smooth = smoothness_score(r_wr_hist)
        l_leg_smooth = smoothness_score(l_an_hist)
        r_leg_smooth = smoothness_score(r_an_hist)

        if frame_count < DETECTION_WINDOW and not locked:
            instr = "Move both arms and legs"
            (tw, _), _ = cv2.getTextSize(instr, FONT, 0.8, 2)
            cv2.putText(frame, instr, ((w - tw)//2, 40),
                        FONT, 0.8, (0,255,255), 2)

        if frame_count >= DETECTION_WINDOW and not locked:
            left_total = left_arm_motion + left_leg_motion
            right_total = right_arm_motion + right_leg_motion

            diff = abs(left_total - right_total)
            confidence = diff / max(left_total, right_total, 1)

            if left_total < right_total * ASYMMETRY_THRESHOLD:
                detected = "LEFT side affected"
            elif right_total < left_total * ASYMMETRY_THRESHOLD:
                detected = "RIGHT side affected"
            else:
                detected = "No clear side affected"

            if confidence >= LOCK_CONFIDENCE and detected != "No clear side affected":
                affected_side = detected
                locked = True
            else:
                affected_side = detected

        cv2.putText(frame, f"Affected side: {affected_side}",
                    (10, 30), FONT, 0.8, (255,255,0), 2)

        cv2.putText(frame, f"Confidence: {confidence:.2f}",
                    (10, 60), FONT, 0.7, (255,255,255), 2)

        bar_w = int(250 * min(confidence / LOCK_CONFIDENCE, 1.0))
        cv2.rectangle(frame, (10, 80), (260, 100), (100,100,100), 2)
        cv2.rectangle(frame, (10, 80), (10 + bar_w, 100), (0,255,0), -1)

        y = 130
        for txt in [
            f"L arm smooth: {l_arm_smooth:.2f}",
            f"R arm smooth: {r_arm_smooth:.2f}",
            f"L leg smooth: {l_leg_smooth:.2f}",
            f"R leg smooth: {r_leg_smooth:.2f}"
        ]:
            cv2.putText(frame, txt, (10, y),
                        FONT, 0.6, (200,255,200), 2)
            y += 22

        if locked:
            draw_right_text(frame, "DETECTION LOCKED", 30, 0.9, (0,255,0), 2)

        for p in [l_ear, r_ear, l_sh, r_sh, l_el, r_el, l_wr, r_wr,
                  l_hip, r_hip, l_kn, r_kn, l_an, r_an]:
            cv2.circle(frame, p, 4, (220,220,220), -1)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
