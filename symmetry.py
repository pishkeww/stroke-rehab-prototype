import cv2
import mediapipe as mp
from screeninfo import get_monitors

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

FONT = cv2.FONT_HERSHEY_SIMPLEX

monitor = get_monitors()[0]
screen_w, screen_h = monitor.width, monitor.height

WINDOW_NAME = "Stroke Rehab – Affected Side Detection"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, screen_w, screen_h)

left_arm_motion = right_arm_motion = 0
left_leg_motion = right_leg_motion = 0

prev = {}
frame_count = 0

DETECTION_WINDOW = int(7 * fps)
ASYMMETRY_THRESHOLD = 0.80
LOCK_CONFIDENCE = 0.35

affected_side = "Detecting..."
locked = False
confidence = 0.0

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

        l_wr = P(mp_pose.PoseLandmark.LEFT_WRIST)
        r_wr = P(mp_pose.PoseLandmark.RIGHT_WRIST)
        l_an = P(mp_pose.PoseLandmark.LEFT_ANKLE)
        r_an = P(mp_pose.PoseLandmark.RIGHT_ANKLE)

        if "l_wr" in prev and not locked:
            left_arm_motion += abs(l_wr[1] - prev["l_wr"][1])
            right_arm_motion += abs(r_wr[1] - prev["r_wr"][1])
            left_leg_motion += abs(l_an[1] - prev["l_an"][1])
            right_leg_motion += abs(r_an[1] - prev["r_an"][1])

        prev["l_wr"], prev["r_wr"] = l_wr, r_wr
        prev["l_an"], prev["r_an"] = l_an, r_an

        if not locked:
            frame_count += 1

        if frame_count < DETECTION_WINDOW and not locked:
            instr = "Please move BOTH arms and legs"
            (tw, _), _ = cv2.getTextSize(instr, FONT, 0.9, 2)
            cv2.putText(frame, instr,
                        ((w - tw)//2, 50),
                        FONT, 0.9, (0,255,255), 2)

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

        cv2.putText(frame,
                    f"Affected side: {affected_side}",
                    (10, 30),
                    FONT, 0.9, (255,255,0), 2)

        cv2.putText(frame,
                    f"Confidence: {confidence:.2f}",
                    (10, 65),
                    FONT, 0.7, (255,255,255), 2)

        bar_x, bar_y = 10, 90
        bar_width = int(300 * min(confidence / LOCK_CONFIDENCE, 1.0))

        cv2.rectangle(frame,
                      (bar_x, bar_y),
                      (bar_x + 300, bar_y + 20),
                      (100,100,100), 2)
        cv2.rectangle(frame,
                      (bar_x, bar_y),
                      (bar_x + bar_width, bar_y + 20),
                      (0,255,0), -1)

        if locked:
            draw_right_text(frame,
                            "DETECTION LOCKED",
                            30, 0.8, (0,255,0), 2)

        for p in [l_wr, r_wr, l_an, r_an]:
            cv2.circle(frame, p, 6, (0,255,0), -1)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
