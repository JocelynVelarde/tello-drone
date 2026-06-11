"""
color_calibrator.py

Live HSV calibrator for the Tello forward camera.
Connects to the drone, streams video, and lets you dial in HSV ranges
for purple, yellow, and blue (one at a time) using trackbars.

Controls:
    n  -> next color (cycles purple -> yellow -> blue -> done)
    s  -> save current color's range to colors.json
    r  -> reset current color's trackbars to defaults
    q  -> quit without saving the in-progress color

Output:
    colors.json  -- dict of {color_name: {"lower": [H,S,V], "upper": [H,S,V]}}
"""

import json
import cv2
import numpy as np
from djitellopy import Tello

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
COLORS_TO_CALIBRATE = ["purple", "yellow", "blue"]
OUTPUT_FILE = "colors.json"

# Sensible starting ranges (rough guesses; you'll tune them with the trackbars)
DEFAULT_RANGES = {
    "purple": {"lower": [125, 50, 50],  "upper": [160, 255, 255]},
    "yellow": {"lower": [20,  100, 100], "upper": [35,  255, 255]},
    "blue":   {"lower": [95,  100, 50],  "upper": [125, 255, 255]},
}

WINDOW_CTRL = "Controls"
WINDOW_VIDEO = "Tello Live (original | mask | result)"

# ---------------------------------------------------------------------------
# Trackbar helpers
# ---------------------------------------------------------------------------
def _noop(_):
    """OpenCV trackbars require a callback; we read values manually instead."""
    pass


def create_trackbars(initial):
    """Create the 6 HSV trackbars in the controls window."""
    cv2.namedWindow(WINDOW_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_CTRL, 400, 300)

    # Hue is 0-179 in OpenCV, S and V are 0-255
    cv2.createTrackbar("H low",  WINDOW_CTRL, initial["lower"][0], 179, _noop)
    cv2.createTrackbar("H high", WINDOW_CTRL, initial["upper"][0], 179, _noop)
    cv2.createTrackbar("S low",  WINDOW_CTRL, initial["lower"][1], 255, _noop)
    cv2.createTrackbar("S high", WINDOW_CTRL, initial["upper"][1], 255, _noop)
    cv2.createTrackbar("V low",  WINDOW_CTRL, initial["lower"][2], 255, _noop)
    cv2.createTrackbar("V high", WINDOW_CTRL, initial["upper"][2], 255, _noop)


def read_trackbars():
    """Read the current HSV bounds from the trackbars."""
    lower = np.array([
        cv2.getTrackbarPos("H low",  WINDOW_CTRL),
        cv2.getTrackbarPos("S low",  WINDOW_CTRL),
        cv2.getTrackbarPos("V low",  WINDOW_CTRL),
    ])
    upper = np.array([
        cv2.getTrackbarPos("H high", WINDOW_CTRL),
        cv2.getTrackbarPos("S high", WINDOW_CTRL),
        cv2.getTrackbarPos("V high", WINDOW_CTRL),
    ])
    return lower, upper


def set_trackbars(values):
    """Push a {'lower':[...], 'upper':[...]} dict back onto the trackbars."""
    cv2.setTrackbarPos("H low",  WINDOW_CTRL, values["lower"][0])
    cv2.setTrackbarPos("S low",  WINDOW_CTRL, values["lower"][1])
    cv2.setTrackbarPos("V low",  WINDOW_CTRL, values["lower"][2])
    cv2.setTrackbarPos("H high", WINDOW_CTRL, values["upper"][0])
    cv2.setTrackbarPos("S high", WINDOW_CTRL, values["upper"][1])
    cv2.setTrackbarPos("V high", WINDOW_CTRL, values["upper"][2])


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_label(img, text, color=(0, 255, 0)):
    """Top-left status label on an image."""
    cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2, cv2.LINE_AA)


def make_composite(original, mask, result):
    """Stack original | mask (as 3ch) | result side by side."""
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return np.hstack([original, mask_bgr, result])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- Connect to the drone ----------------------------------------------
    print("Connecting to Tello...")
    tello = Tello()
    tello.connect()
    print(f"Connected. Battery: {tello.get_battery()}%")

    tello.streamon()
    frame_reader = tello.get_frame_read()

    # --- Set up windows ----------------------------------------------------
    cv2.namedWindow(WINDOW_VIDEO, cv2.WINDOW_NORMAL)
    create_trackbars(DEFAULT_RANGES[COLORS_TO_CALIBRATE[0]])

    saved = {}                  # final dict we'll write to JSON
    idx = 0                     # which color we're currently calibrating

    print("\n" + "=" * 60)
    print("CALIBRATOR CONTROLS")
    print("  s  ->  save current color and move to next")
    print("  n  ->  skip current color (do NOT save) and move to next")
    print("  r  ->  reset trackbars to default for this color")
    print("  q  ->  quit (writes whatever you've saved so far)")
    print("=" * 60 + "\n")
    print(f"[1/{len(COLORS_TO_CALIBRATE)}] Calibrating: "
          f"{COLORS_TO_CALIBRATE[idx].upper()}")
    print("Hold a sheet of this color in front of the camera and adjust "
          "the trackbars\nuntil only that color is white in the mask.")

    try:
        while True:
            frame = frame_reader.frame
            if frame is None:
                continue

            # Tello frames arrive as RGB; convert to BGR for OpenCV display
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (480, 360))

            # Apply the current trackbar HSV range
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower, upper = read_trackbars()
            mask = cv2.inRange(hsv, lower, upper)

            # Light cleanup so small noise specks don't clutter the view
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            result = cv2.bitwise_and(frame, frame, mask=mask)

            # Status label on the original frame
            current_color = COLORS_TO_CALIBRATE[idx]
            draw_label(frame,
                       f"[{idx + 1}/{len(COLORS_TO_CALIBRATE)}] {current_color.upper()}"
                       f"  bat:{tello.get_battery()}%")

            composite = make_composite(frame, mask, result)
            cv2.imshow(WINDOW_VIDEO, composite)

            # --- Handle keyboard ---------------------------------------------
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Quit requested.")
                break

            elif key == ord('r'):
                set_trackbars(DEFAULT_RANGES[current_color])
                print(f"  reset {current_color} trackbars to defaults")

            elif key == ord('n'):
                print(f"  skipped {current_color} (not saved)")
                idx += 1
                if idx >= len(COLORS_TO_CALIBRATE):
                    print("Reached end of color list.")
                    break
                set_trackbars(DEFAULT_RANGES[COLORS_TO_CALIBRATE[idx]])
                print(f"\n[{idx + 1}/{len(COLORS_TO_CALIBRATE)}] "
                      f"Calibrating: {COLORS_TO_CALIBRATE[idx].upper()}")

            elif key == ord('s'):
                saved[current_color] = {
                    "lower": lower.tolist(),
                    "upper": upper.tolist(),
                }
                print(f"  saved {current_color}: "
                      f"lower={lower.tolist()} upper={upper.tolist()}")
                idx += 1
                if idx >= len(COLORS_TO_CALIBRATE):
                    print("All colors calibrated.")
                    break
                set_trackbars(DEFAULT_RANGES[COLORS_TO_CALIBRATE[idx]])
                print(f"\n[{idx + 1}/{len(COLORS_TO_CALIBRATE)}] "
                      f"Calibrating: {COLORS_TO_CALIBRATE[idx].upper()}")

    finally:
        # --- Persist whatever was saved -----------------------------------
        if saved:
            with open(OUTPUT_FILE, "w") as f:
                json.dump(saved, f, indent=2)
            print(f"\nWrote {len(saved)} color(s) to {OUTPUT_FILE}")
        else:
            print("\nNothing was saved.")

        # --- Clean shutdown -----------------------------------------------
        cv2.destroyAllWindows()
        try:
            tello.streamoff()
        except Exception:
            pass
        tello.end()


if __name__ == "__main__":
    main()