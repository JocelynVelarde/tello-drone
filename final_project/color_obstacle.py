"""
color_obstacle_detector.py  --  minimal, in-flight, executes maneuvers

Hover and react to colored papers:
    yellow -> arc right
    blue   -> arc left
    purple -> descend 30 cm

Controls:
    q -> land and quit (focus must be on the OpenCV window)
    Ctrl+C in terminal also lands.
"""

import json
import time
import cv2
import numpy as np
from djitellopy import Tello

COLORS_FILE = "colors.json"
MIN_AREA = 5000          # ignore small blobs
CONFIRM_FRAMES = 3       # must see same color this many frames in a row
MAX_FLIGHT_SEC = 90      # hard safety timeout

DRAW = {
    "yellow": (0, 255, 255),
    "blue":   (255, 80, 0),
    "purple": (200, 0, 200),
}


def detect(hsv, lower, upper):
    """Return (box, area) for the largest blob of this color, or None."""
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < MIN_AREA:
        return None
    return cv2.boundingRect(c), area


def react(tello, color):
    """Execute the maneuver for the given color, then return to hover."""
    print(f"  -> executing {color}")
    try:
        if color == "yellow":
            # Arc RIGHT: midpoint at (60, -20), endpoint at (80, -80)
            tello.curve_xyz_speed(60, -20, 0, 80, -80, 0, 30)
        elif color == "blue":
            # Arc LEFT: midpoint at (60, 20), endpoint at (80, 80)
            tello.curve_xyz_speed(60, 20, 0, 80, 80, 0, 30)
        elif color == "purple":
            tello.move_down(30)
    except Exception as e:
        # Don't kill the whole flight if a single maneuver fails
        print(f"  maneuver failed: {e}")
    time.sleep(0.5)


def main():
    with open(COLORS_FILE) as f:
        colors = json.load(f)

    tello = Tello()
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")

    tello.streamon()
    time.sleep(2)
    reader = tello.get_frame_read()
    time.sleep(2)

    cv2.namedWindow("tello", cv2.WINDOW_NORMAL)

    streak_color = None
    streak_count = 0
    flying = False
    start = None

    try:
        print("Taking off...")
        tello.takeoff()
        flying = True
        start = time.time()

        while True:
            if time.time() - start > MAX_FLIGHT_SEC:
                print("Timeout. Landing.")
                break

            frame = reader.frame
            if frame is None:
                cv2.waitKey(5)
                continue

            frame = cv2.resize(frame, (480, 360))
            # Tello streams frames as RGB; the calibrator did this same
            # conversion before sampling HSV, so we MUST do it here too
            # for the saved ranges to be valid.
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            best_color = None
            best_area = 0
            best_box = None
            for name in ("yellow", "blue", "purple"):
                r = detect(hsv, colors[name]["lower"], colors[name]["upper"])
                if r is not None:
                    box, area = r
                    if area > best_area:
                        best_area = area
                        best_color = name
                        best_box = box

            if best_box is not None:
                x, y, w, h = best_box
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              DRAW[best_color], 3)
                cv2.putText(frame, best_color, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            DRAW[best_color], 2)

            cv2.imshow("tello", frame)

            if best_color == streak_color and best_color is not None:
                streak_count += 1
            else:
                streak_color = best_color
                streak_count = 1 if best_color else 0

            if streak_count >= CONFIRM_FRAMES:
                react(tello, streak_color)
                streak_color = None
                streak_count = 0

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Ctrl+C. Landing.")
    finally:
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