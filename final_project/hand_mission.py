"""
hand_mission.py  --  Tello hand-gesture control (simple, no reporting)

Take off, then steer by finger count (each held CONFIRM_FRAMES frames):
    2 or 3 fingers -> move forward 20 cm
    5 fingers      -> move back    30 cm
Other counts ignored. Land after TARGET_MOVES gestures, or press 'q'.

pip install djitellopy opencv-python numpy mediapipe
Requires hand_landmarker.task next to this file
(https://ai.google.dev/edge/mediapipe -> Hand Landmarker).

Keys:  q -> land and quit     d -> toggle angle debug overlay
"""

import time
import math
import threading
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from djitellopy import Tello

MODEL_PATH       = "hand_landmarker.task"
CONFIRM_FRAMES   = 2      # frames a gesture must persist before it fires
COOLDOWN_SEC     = 1.0    # min time between gesture commands
TARGET_MOVES     = 4      # land after this many gestures
MAX_FLIGHT_SEC   = 120
WINDOW = "Tello Hand Mission"


class VideoRecorder(threading.Thread):
    """Writes the clean drone feed to an .mp4 from its own thread, so the
    recording stays smooth even while a blocking move freezes the main loop."""
    def __init__(self, reader, path, fps=30, size=(640, 480)):
        super().__init__(daemon=True)
        self.reader, self.path, self.fps, self.size = reader, path, fps, size
        self._stop_evt = threading.Event()
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        self.ok = self.writer.isOpened()
        if not self.ok:
            print("WARNING: could not open video writer; not recording.")

    def run(self):
        if not self.ok:
            return
        interval = 1.0 / self.fps
        while not self._stop_evt.is_set():
            t = time.time()
            frame = self.reader.frame
            if frame is not None and getattr(frame, "size", 0) > 0:
                self.writer.write(cv2.cvtColor(cv2.resize(frame, self.size),
                                               cv2.COLOR_RGB2BGR))
            dt = time.time() - t
            if dt < interval:
                time.sleep(interval - dt)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2)
        if self.ok:
            self.writer.release()

# finger count -> (command, distance). Other counts ignored.
ACTIONS = {2: ("front", 30), 3: ("front", 30), 5: ("back", 30)}

FINGER_JOINTS = {"thumb": (2, 3, 4), "index": (5, 6, 8), "middle": (9, 10, 12),
                 "ring": (13, 14, 16), "pinky": (17, 18, 20)}
THRESH = {"thumb": 150.0, "index": 160.0, "middle": 160.0, "ring": 160.0, "pinky": 160.0}
CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),
        (11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]


def angle(lms, a, b, c):
    pa = np.array([lms[a].x, lms[a].y, lms[a].z])
    pb = np.array([lms[b].x, lms[b].y, lms[b].z])
    pc = np.array([lms[c].x, lms[c].y, lms[c].z])
    ba, bc = pa - pb, pc - pb
    if np.linalg.norm(ba) < 1e-6 or np.linalg.norm(bc) < 1e-6:
        return 180.0
    cos = np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)), -1, 1)
    return math.degrees(math.acos(cos))


def count_fingers(lms, frame, debug):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lms]
    for a, b in CONN:
        cv2.line(frame, pts[a], pts[b], (0, 200, 200), 2)
    n = 0
    for name, (a, b, c) in FINGER_JOINTS.items():
        ext = angle(lms, a, b, c) >= THRESH[name]
        n += ext
        if debug:
            cv2.circle(frame, pts[c], 7, (0, 255, 0) if ext else (0, 0, 255), -1)
    return n


def move(tello, action, dist, fingers):
    print(f"{fingers} fingers -> move {action} {dist}")
    try:
        if action == "front":
            tello.move_forward(dist)
        elif action == "back":
            tello.move_back(dist)
    except Exception as e:
        print("move failed:", e)


def main():
    base = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base, running_mode=mp_vision.RunningMode.VIDEO, num_hands=1)
    detector = mp_vision.HandLandmarker.create_from_options(opts)

    tello = Tello()
    tello.connect()
    print("Battery:", tello.get_battery(), "%")
    tello.RESPONSE_TIMEOUT = 10
    try:
        tello.set_video_resolution(Tello.RESOLUTION_480P)
        tello.set_video_fps(Tello.FPS_30)
    except Exception:
        pass

    tello.streamon()
    reader = tello.get_frame_read()
    t0 = time.time()
    while reader.frame is None or getattr(reader.frame, "size", 0) == 0:
        if time.time() - t0 > 12:
            raise SystemExit("No video — check Wi-Fi/battery.")
        time.sleep(0.1)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    rec_path = time.strftime("hand_flight_%Y%m%d_%H%M%S.mp4")
    recorder = VideoRecorder(reader, rec_path)
    recorder.start()
    print("Recording to", rec_path)
    streak_count, streak_n, last_t, moves = None, 0, 0.0, 0
    debug, flying = True, False

    try:
        tello.takeoff()
        flying = True
        start = time.time()
        gt0 = time.time()

        while True:
            if time.time() - start > MAX_FLIGHT_SEC:
                break

            frame = reader.frame
            if frame is None:
                cv2.waitKey(5); continue
            frame = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_RGB2BGR)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(mp_img, int((time.time() - gt0) * 1000))

            fingers = None
            if result.hand_landmarks:
                fingers = count_fingers(result.hand_landmarks[0], frame, debug)

            if fingers == streak_count and fingers is not None:
                streak_n += 1
            else:
                streak_count, streak_n = fingers, 1

            if (streak_n == CONFIRM_FRAMES and time.time() - last_t > COOLDOWN_SEC
                    and fingers in ACTIONS):
                action, dist = ACTIONS[fingers]
                move(tello, action, dist, fingers)
                last_t = time.time()
                moves += 1
                if moves >= TARGET_MOVES:
                    break

            cv2.putText(frame, f"fingers {fingers if fingers is not None else '-'}  "
                        f"moves {moves}/{TARGET_MOVES}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                debug = not debug

    except KeyboardInterrupt:
        pass
    finally:
        recorder.stop()
        if flying:
            try:
                tello.land()
            except Exception:
                tello.emergency()
        detector.close()
        cv2.destroyAllWindows()
        try:
            tello.streamoff()
        except Exception:
            pass
        tello.end()


if __name__ == "__main__":
    main()