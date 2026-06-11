"""
color_mission.py  --  Tello color-sheet movement (simple, no reporting)

Take off, react to colored sheets:
    yellow -> move RIGHT 30 cm
    blue   -> move LEFT  30 cm
    purple -> move UP    30 cm
A color must persist CONFIRM_FRAMES frames before its move fires.
Land after TARGET_TRIGGERS detections, or press 'q' to land early.

pip install djitellopy opencv-python numpy
Requires colors.json next to this file.
"""

import time
import json
import threading
import cv2
import numpy as np
from djitellopy import Tello

COLORS_FILE      = "colors.json"
TARGET_TRIGGERS  = 5      # land after this many color detections
CONFIRM_FRAMES   = 3      # frames a color must persist before reacting
MOVE_CM          = 30     # move distance (Tello min 20)
MIN_AREA         = 5000   # ignore color blobs smaller than this
MAX_FLIGHT_SEC   = 120
COLOR_ORDER      = ("yellow", "blue", "purple")
DRAW = {"yellow": (0, 255, 255), "blue": (255, 80, 0), "purple": (200, 0, 200)}
WINDOW = "Tello Color Mission"


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


def detect_color(hsv, lower, upper):
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA:
        return None
    return cv2.boundingRect(c)


def react(tello, color):
    print(f"{color} -> move {color and {'yellow':'right','blue':'left','purple':'up'}[color]} {MOVE_CM}")
    try:
        if color == "yellow":
            tello.move_right(MOVE_CM)
        elif color == "blue":
            tello.move_left(MOVE_CM)
        elif color == "purple":
            tello.move_up(MOVE_CM)
    except Exception as e:
        print("move failed:", e)


def main():
    with open(COLORS_FILE) as f:
        colors = json.load(f)

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
    rec_path = time.strftime("color_flight_%Y%m%d_%H%M%S.mp4")
    recorder = VideoRecorder(reader, rec_path)
    recorder.start()
    print("Recording to", rec_path)
    streak_color, streak_n, triggers = None, 0, 0
    flying = False

    try:
        tello.takeoff()
        flying = True
        start = time.time()

        while True:
            if time.time() - start > MAX_FLIGHT_SEC:
                break

            frame = reader.frame
            if frame is None:
                cv2.waitKey(5); continue
            frame = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            best, best_area, best_box = None, 0, None
            for name in COLOR_ORDER:
                box = detect_color(hsv, colors[name]["lower"], colors[name]["upper"])
                if box and box[2] * box[3] > best_area:
                    best, best_area, best_box = name, box[2] * box[3], box

            if best_box:
                x, y, w, h = best_box
                cv2.rectangle(frame, (x, y), (x+w, y+h), DRAW[best], 3)
                cv2.putText(frame, best, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, DRAW[best], 2)

            if best == streak_color and best is not None:
                streak_n += 1
            else:
                streak_color, streak_n = best, (1 if best else 0)

            if streak_n >= CONFIRM_FRAMES:
                react(tello, streak_color)
                triggers += 1
                streak_color, streak_n = None, 0
                if triggers >= TARGET_TRIGGERS:
                    break

            cv2.putText(frame, f"colors {triggers}/{TARGET_TRIGGERS}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(WINDOW, frame)
            if (cv2.waitKey(5) & 0xFF) == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        recorder.stop()
        if flying:
            try:
                tello.land()
            except Exception:
                tello.emergency()
        cv2.destroyAllWindows()
        try:
            tello.streamoff()
        except Exception:
            pass
        tello.end()


if __name__ == "__main__":
    main()