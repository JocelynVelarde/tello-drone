"""
matrix_test.py

Standalone test for the 8x8 RGB LED matrix on the Tello Talent
expansion board. No flight -- the drone stays on the ground.

Cycles through:
  1. Solid yellow  (paper color)
  2. Solid blue    (paper color)
  3. Solid purple  (paper color)
  4. Digit "1"     (finger count)
  5. Digit "2"     (finger count)
  6. Digit "5"     (finger count)
  7. Clear / off

Each step lasts a few seconds so you can confirm visually.

Controls:
    q -> quit (in the small status window)
"""

import time
import cv2
import numpy as np
from djitellopy import Tello

# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------
# Custom 8x8 bitmaps for digits 1, 2, 5.
# Each is 64 characters (8 rows x 8 cols), row-major.
# We use 'b' for "on" pixels (blue) and '0' for off. We'll override the
# color of each digit at draw time by passing a different palette char.
#
# Visual reference -- each '#' means an "on" pixel, '.' means off:
#
# DIGIT_1                DIGIT_2                DIGIT_5
# . . . # # . . .        . # # # # # # .        . # # # # # # .
# . . # # # . . .        # # . . . . # #        . # . . . . . .
# . . . # # . . .        . . . . . # # .        . # . . . . . .
# . . . # # . . .        . . . # # # . .        . # # # # # # .
# . . . # # . . .        . . # # # . . .        . . . . . . # .
# . . . # # . . .        . # # . . . . .        . . . . . . # .
# . . . # # . . .        # # . . . . . .        # # . . . . # #
# . # # # # # # .        # # # # # # # #        . # # # # # # .

DIGIT_BITMAPS = {
    1: (
        "0001100000011100000011000000110000001100000011000000110000111111"
    ),
    2: (
        "01111110110000010000001000001100000110000011000001100000111111111"[:64]
    ),
    5: (
        "0111111101000000010000000111111000000010000000100110001000111100"
    ),
}

# Sanity: every bitmap must be exactly 64 chars
for d, b in DIGIT_BITMAPS.items():
    assert len(b) == 64, f"digit {d} bitmap is {len(b)} chars, expected 64"


def matrix_solid_rgb(tello, r, g, b):
    """Light the whole matrix with one RGB color (each value 0..255)."""
    tello.send_expansion_command(f"mled g {r} {g} {b}")


def matrix_clear(tello):
    """Turn all LEDs off."""
    tello.send_expansion_command("mled sc")


def matrix_digit(tello, digit, color_char="b"):
    """
    Display a single digit (1, 2, or 5) using our custom bitmap.

    Args:
        digit:      one of 1, 2, 5
        color_char: 'r' red, 'b' blue, 'p' purple, '0' off,
                    or uppercase R/B/P for brighter. The matrix firmware
                    only supports these few colors for bitmap mode.
    """
    if digit not in DIGIT_BITMAPS:
        raise ValueError(f"no bitmap for digit {digit}")

    # Replace every 'b' in our template with the chosen color char
    template = DIGIT_BITMAPS[digit]
    bitmap = template.replace("b", color_char)

    # The firmware command is "mled g <bitmap>" for a custom pattern
    tello.send_expansion_command(f"mled g {bitmap}")


def show_status(text, color=(255, 255, 255)):
    """Tiny OpenCV window showing what the matrix is currently doing,
    plus a 'q' shortcut to bail out cleanly."""
    img = np.zeros((200, 600, 3), dtype=np.uint8)
    cv2.putText(img, text, (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)
    cv2.putText(img, "press q to quit", (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.imshow("matrix_test", img)


# ---------------------------------------------------------------------------
# Main test sequence
# ---------------------------------------------------------------------------
def main():
    print("Connecting to Tello...")
    tello = Tello()
    tello.connect()
    print(f"Connected. Battery: {tello.get_battery()}%")

    # Start clean
    matrix_clear(tello)
    time.sleep(0.5)

    # (label, action, duration_seconds)
    sequence = [
        ("YELLOW paper",     lambda: matrix_solid_rgb(tello, 255, 255, 0), 2.5),
        ("BLUE paper",       lambda: matrix_solid_rgb(tello, 0, 80, 255),  2.5),
        ("PURPLE paper",     lambda: matrix_solid_rgb(tello, 180, 0, 220), 2.5),
        ("digit 1 (blue)",   lambda: matrix_digit(tello, 1, "b"),          2.5),
        ("digit 2 (red)",    lambda: matrix_digit(tello, 2, "r"),          2.5),
        ("digit 5 (purple)", lambda: matrix_digit(tello, 5, "p"),          2.5),
        ("clear",            lambda: matrix_clear(tello),                  1.5),
    ]

    cv2.namedWindow("matrix_test")

    try:
        for label, action, dur in sequence:
            print(f"-> {label}")
            action()
            show_status(label)

            # During each step, poll the keyboard so 'q' still works.
            start = time.time()
            while time.time() - start < dur:
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    raise KeyboardInterrupt
        print("Done.")

    except KeyboardInterrupt:
        print("Quit requested.")

    finally:
        matrix_clear(tello)
        cv2.destroyAllWindows()
        tello.end()
        print("Shut down cleanly.")


if __name__ == "__main__":
    main()