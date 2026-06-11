"""
finger_detector.py  --  angle-based + MediaPipe Tasks API

Standalone finger counter using the Tello forward camera and the
HandLandmarker (Tasks API). Counts via joint angles so the result is
robust to hand tilt and rotation.

Controls:
    q  -> quit
    d  -> toggle angle debug overlay
"""

import time
import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from djitellopy import Tello

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = "hand_landmarker.task"

# For each finger, we measure the angle at the MIDDLE landmark of three.
# A near-180-degree angle means the finger is straight (extended);
# a bent angle means the finger is curled (closed).
#
# Landmark map (MediaPipe Hands, same 21 points):
#   Thumb : CMC=1,  MCP=2,  IP=3,   tip=4
#   Index : MCP=5,  PIP=6,  DIP=7,  tip=8
#   Middle: MCP=9,  PIP=10, DIP=11, tip=12
#   Ring  : MCP=13, PIP=14, DIP=15, tip=16
#   Pinky : MCP=17, PIP=18, DIP=19, tip=20
FINGER_JOINTS = {
    # name      (a,   b,   c)   -- b is the vertex of the angle
    "thumb":  (2,   3,   4),    # MCP -- IP  -- tip
    "index":  (5,   6,   8),    # MCP -- PIP -- tip
    "middle": (9,   10,  12),
    "ring":   (13,  14,  16),
    "pinky":  (17,  18,  20),
}

# A finger counts as "extended" if its angle is >= this threshold (degrees).
# The thumb threshold is lower because the thumb never fully straightens.
EXTENDED_THRESHOLD_DEG = {
    "thumb":  150.0,
    "index":  160.0,
    "middle": 160.0,
    "ring":   160.0,
    "pinky":  160.0,
}

# Skeleton connections (pairs of landmark indices) for drawing the hand.
# Replaces mp_hands.HAND_CONNECTIONS from the old API.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (5, 9), (9, 10), (10, 11), (11, 12),    # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (17, 18), (18, 19), (19, 20), # pinky
    (0, 17),                                 # palm base to pinky base
]

WINDOW_NAME = "Tello Finger Detector"


# ---------------------------------------------------------------------------
# Angle math
# ---------------------------------------------------------------------------
def joint_angle_deg(landmarks, a_idx, b_idx, c_idx):
    """
    Return the angle (in degrees) at joint b, formed by the vectors
    b->a and b->c. Uses 3D coordinates from MediaPipe (x, y, z), which
    gives orientation-invariant results.

    A straight finger -> angle near 180.
    A bent finger     -> angle drops toward 0 as the bend deepens.
    """
    a = np.array([landmarks[a_idx].x, landmarks[a_idx].y, landmarks[a_idx].z])
    b = np.array([landmarks[b_idx].x, landmarks[b_idx].y, landmarks[b_idx].z])
    c = np.array([landmarks[c_idx].x, landmarks[c_idx].y, landmarks[c_idx].z])

    ba = a - b
    bc = c - b

    # Guard against zero-length vectors
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-6 or nbc < 1e-6:
        return 180.0  # treat as "straight" to avoid spurious closed counts

    cos_angle = np.dot(ba, bc) / (nba * nbc)
    cos_angle = max(-1.0, min(1.0, cos_angle))  # clamp for numerical safety
    return math.degrees(math.acos(cos_angle))


def finger_states(landmarks):
    """
    Compute (name, angle_deg, is_extended) for each of the five fingers.
    `landmarks` is a list of 21 NormalizedLandmark objects.
    """
    out = []
    for name, (a, b, c) in FINGER_JOINTS.items():
        angle = joint_angle_deg(landmarks, a, b, c)
        is_ext = angle >= EXTENDED_THRESHOLD_DEG[name]
        out.append((name, angle, is_ext))
    return out


def count_fingers(landmarks):
    """Return how many fingers are extended (0..5)."""
    return sum(1 for _, _, is_ext in finger_states(landmarks) if is_ext)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_landmarks(frame, landmarks):
    """Skeleton + keypoints for one hand."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 200), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)


def draw_angle_debug(frame, landmarks, states):
    """
    Print each finger's angle at its bending joint and color the tip
    green (counted extended) or red (counted closed). Lets you see why
    each finger is being counted the way it is, so you can tune the
    thresholds.
    """
    h, w = frame.shape[:2]

    # `states` is in the same order as FINGER_JOINTS, so we can zip them
    for (name, angle, is_ext), (_, joint_b, tip_idx) in zip(
            states, FINGER_JOINTS.values()):
        color = (0, 255, 0) if is_ext else (0, 0, 255)

        # Angle label near the bending joint
        bx = int(landmarks[joint_b].x * w)
        by = int(landmarks[joint_b].y * h)
        cv2.putText(frame, f"{int(angle)}", (bx + 6, by + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Tip indicator: filled circle, green=open, red=closed
        tx = int(landmarks[tip_idx].x * w)
        ty = int(landmarks[tip_idx].y * h)
        cv2.circle(frame, (tx, ty), 8, color, -1)


def draw_info(frame, count, fps, battery):
    """Top-left big count + top-right status line + bottom hint."""
    h, w = frame.shape[:2]

    label = f"Fingers: {count}" if count is not None else "Fingers: --"
    cv2.rectangle(frame, (5, 5), (260, 70), (0, 0, 0), -1)
    cv2.putText(frame, label, (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.putText(frame, f"FPS:{fps:4.1f}  BAT:{battery}%",
                (w - 230, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, "Press 'q' to quit, 'd' to toggle debug",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- Build the HandLandmarker -----------------------------------------
    # VIDEO mode = synchronous, fed monotonically increasing timestamps.
    # Right fit for a per-frame loop like this one.
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    # --- Connect to drone (no takeoff) ------------------------------------
    print("Connecting to Tello...")
    tello = Tello()
    tello.connect()
    print(f"Connected. Battery: {tello.get_battery()}%")

    tello.streamon()
    frame_reader = tello.get_frame_read()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    last_logged_count = None
    prev_time = time.time()
    fps = 0.0
    debug_overlay = True            # toggle with 'd'
    start_time_ms = int(time.time() * 1000)

    print("\nReady. Hold your hand in front of the drone's forward camera.")
    print("Number next to each joint = angle in degrees.")
    print("Green tip dot = counted OPEN, red = counted CLOSED.\n")

    try:
        while True:
            frame = frame_reader.frame
            if frame is None:
                continue

            # Tello frames are RGB; convert to BGR for OpenCV display.
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (640, 480))

            # Tasks API expects an mp.Image in SRGB (RGB) format.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # VIDEO mode requires a strictly-increasing timestamp in ms.
            timestamp_ms = int(time.time() * 1000) - start_time_ms
            result = detector.detect_for_video(mp_image, timestamp_ms)

            count = None
            states = None
            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                states = finger_states(landmarks)
                count = sum(1 for _, _, is_ext in states if is_ext)

                draw_landmarks(frame, landmarks)
                if debug_overlay:
                    draw_angle_debug(frame, landmarks, states)

            # --- Log on change -------------------------------------------
            if count != last_logged_count:
                if count is None:
                    print(f"[{time.strftime('%H:%M:%S')}] no hand detected")
                else:
                    extended = [n for n, _, e in states if e]
                    print(f"[{time.strftime('%H:%M:%S')}] "
                          f"fingers = {count}  extended={extended}")
                last_logged_count = count

            # --- FPS -----------------------------------------------------
            now = time.time()
            dt = now - prev_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            prev_time = now

            draw_info(frame, count, fps, tello.get_battery())
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                debug_overlay = not debug_overlay
                print(f"  debug overlay {'ON' if debug_overlay else 'OFF'}")

    finally:
        detector.close()
        cv2.destroyAllWindows()
        try:
            tello.streamoff()
        except Exception:
            pass
        tello.end()
        print("\nShut down cleanly.")


if __name__ == "__main__":
    main()