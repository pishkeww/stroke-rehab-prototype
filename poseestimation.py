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

good_frames = 0
bad_frames = 0

FONT = cv2.FONT_HERSHEY_SIMPLEX

monitor = get_monitors()[0]
screen_w, screen_h = monitor.width, monitor.height

WINDOW_NAME = "Stroke Rehab – Full System"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, screen_w, screen_h)

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

        head_tilt = abs(l_ear[1] - r_ear[1])
        shoulder_diff = abs(l_sh[1] - r_sh[1])
        elbow_diff = abs(l_el[1] - r_el[1])
        wrist_diff = abs(l_wr[1] - r_wr[1])
        hip_diff = abs(l_hip[1] - r_hip[1])
        knee_diff = abs(l_kn[1] - r_kn[1])
        ankle_diff = abs(l_an[1] - r_an[1])

        left_hand_active = abs(l_wr[1] - l_sh[1]) > 40
        right_hand_active = abs(r_wr[1] - r_sh[1]) > 40

        good = (
            head_tilt < 25 and
            shoulder_diff < 30 and
            hip_diff < 30 and
            knee_diff < 35 and
            ankle_diff < 40
        )

        if good:
            good_frames += 1
            bad_frames = 0
            color = (0, 255, 0)
            label = "GOOD ALIGNMENT"
        else:
            bad_frames += 1
            good_frames = 0
            color = (0, 0, 255)
            label = "ASYMMETRY DETECTED"

        cv2.putText(frame, label, (10, 30), FONT, 0.9, color, 2)

        y = 65
        for txt in [
            f"Head tilt: {head_tilt}",
            f"Shoulder diff: {shoulder_diff}",
            f"Elbow diff: {elbow_diff}",
            f"Wrist diff: {wrist_diff}",
            f"Hip diff: {hip_diff}",
            f"Knee diff: {knee_diff}",
            f"Ankle diff: {ankle_diff}",
            f"Left hand active: {left_hand_active}",
            f"Right hand active: {right_hand_active}"
        ]:
            cv2.putText(frame, txt, (10, y), FONT, 0.6, color, 2)
            y += 24

        joints = [
            l_ear, r_ear, l_sh, r_sh, l_el, r_el, l_wr, r_wr,
            l_hip, r_hip, l_kn, r_kn, l_an, r_an
        ]

        for p in joints:
            cv2.circle(frame, p, 4, (220, 220, 220), -1)

        links = [
            (l_sh, r_sh),
            (l_sh, l_el), (l_el, l_wr),
            (r_sh, r_el), (r_el, r_wr),
            (l_sh, l_hip), (r_sh, r_hip),
            (l_hip, r_hip),
            (l_hip, l_kn), (l_kn, l_an),
            (r_hip, r_kn), (r_kn, r_an)
        ]

        for a, b in links:
            cv2.line(frame, a, b, color, 2)

        good_time = good_frames / fps
        bad_time = bad_frames / fps

        cv2.putText(frame,
                    f"Good: {good_time:.1f}s  Bad: {bad_time:.1f}s",
                    (10, h - 20),
                    FONT, 0.7, color, 2)

        if bad_time > 180:
            warning = "WARNING: Persistent asymmetry"
            (tw, _), _ = cv2.getTextSize(warning, FONT, 1, 3)
            cv2.putText(frame, warning,
                        ((w - tw)//2, 60),
                        FONT, 1, (0, 0, 255), 3)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
