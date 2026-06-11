"""
mission_a_to_b.py  --  integrated vision-guided Tello mission (A -> B)

Mission logic (state machine):

    Phase 1  COLOR
        Take off at A and react to colored sheets held in front of the camera
        with straight moves (no arcing):
            yellow  -> move RIGHT 30 cm
            blue    -> move LEFT  30 cm
            purple  -> move UP    30 cm
        A color must be the dominant blob for COLOR_CONFIRM_FRAMES consecutive
        frames before its move fires. After MIN_COLOR_TRIGGERS (>=5) successful
        color reactions, the drone hovers and advances to the gesture zone.

    Phase 2  GESTURE
        Run the MediaPipe HandLandmarker finger counter and steer the drone by
        finger count, each held for GESTURE_CONFIRM_FRAMES (>=3) frames:
            2 or 3 fingers -> move forward 20 cm
            5 fingers      -> move back    30 cm
        Other counts (0, 1, 4) are ignored. A cooldown prevents repeats.
        After MIN_GESTURE_MOVES (=3) gestures the drone lands at B. It also
        lands early on 'q', Ctrl-C, or the MAX_FLIGHT_SEC timeout.

On shutdown the script writes, into mission_report/<timestamp>/ :
    - flight_log.csv           every logged event (data table for the report)
    - summary.csv              one-line mission summary
    - battery_vs_time.png      battery telemetry over the run
    - color_triggers.png       count of each color maneuver fired
    - finger_timeline.png      finger count during the gesture phase
    - gesture_moves.png        count of each gesture command executed
    - summary_table.png        rendered summary table for the report

Dependencies:
    pip install djitellopy opencv-python numpy mediapipe matplotlib
    Download the hand model once and place it next to this file:
    hand_landmarker.task  (https://ai.google.dev/edge/mediapipe -> Hand Landmarker)

Controls (focus on an OpenCV window):
    q -> land and quit early        d -> toggle finger-angle debug overlay
"""

import os
import csv
import time
import math
import datetime

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from djitellopy import Tello

# ===========================================================================
# Configuration
# ===========================================================================
COLORS_FILE          = "colors.json"      # produced by the calibrator
MODEL_PATH           = "hand_landmarker.task"

MIN_COLOR_TRIGGERS   = 5      # colored pages to clear before the gesture zone
COLOR_CONFIRM_FRAMES = 3      # frames a color must persist before reacting
GESTURE_CONFIRM_FRAMES = 3    # frames a gesture must persist before it fires

# Finger-count -> (flight command, distance cm) in the gesture phase.
# Counts that are not keys here are IGNORED (no action).
FINGER_ACTIONS = {
    2: ("front", 30),   # 2 fingers -> move forward 20 cm
    3: ("front", 30),   # 3 fingers -> move forward 20 cm
    5: ("back", 30),    # 5 fingers -> move back 30 cm
}
GESTURE_COOLDOWN_SEC = 2.0    # min time between two gesture commands
MIN_GESTURE_MOVES    = 3      # land after this many gesture commands
# Mission ends after MIN_GESTURE_MOVES gestures, or on 'q' / Ctrl-C / timeout.

COLOR_MOVE_CM        = 30     # straight-move distance for each color (Tello min 20)

MIN_AREA             = 5000   # ignore color blobs smaller than this (px)
MAX_FLIGHT_SEC       = 300    # hard safety timeout for the whole mission
BATTERY_SAMPLE_SEC   = 3.0    # how often to log a battery reading

MIN_BATTERY_PCT      = 20     # refuse to fly below this (low batt -> video drops)
RESPONSE_TIMEOUT_SEC = 10     # how long to wait for a command ACK (default 7)
STREAM_WAIT_SEC      = 12     # max wait for the first decodable video frame

REPORT_ROOT          = "mission_report"

DRAW = {"yellow": (0, 255, 255), "blue": (255, 80, 0), "purple": (200, 0, 200)}

# Maneuver order used for color detection priority
COLOR_ORDER = ("yellow", "blue", "purple")

# Finger geometry (MediaPipe 21-landmark hand). b is the vertex of the angle.
FINGER_JOINTS = {
    "thumb":  (2, 3, 4),
    "index":  (5, 6, 8),
    "middle": (9, 10, 12),
    "ring":   (13, 14, 16),
    "pinky":  (17, 18, 20),
}
EXTENDED_THRESHOLD_DEG = {
    "thumb": 150.0, "index": 160.0, "middle": 160.0, "ring": 160.0, "pinky": 160.0,
}
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

WINDOW = "Tello Mission A -> B"


# ===========================================================================
# Mission logger  (collects everything the report needs)
# ===========================================================================
class MissionLog:
    def __init__(self):
        self.t0 = time.time()
        self.events = []                                   # list of dicts
        self.battery_series = []                           # (t, pct)
        self.finger_series = []                            # (t, count)
        self.color_counts = {c: 0 for c in COLOR_ORDER}
        self.color_sequence = []                           # ordered triggers
        self.gesture_counts = {a: 0 for a in ("front", "back")}
        self.gesture_sequence = []                         # ordered gesture moves
        self.landed_on_gesture = False
        self.end_battery = None

    def rel(self):
        return round(time.time() - self.t0, 2)

    def event(self, phase, kind, detail="", battery=None):
        row = {"t": self.rel(), "phase": phase, "event": kind,
               "detail": detail, "battery": battery}
        self.events.append(row)
        print(f"[{row['t']:6.2f}s] {phase:7s} {kind:14s} {detail}")

    def battery(self, pct):
        self.battery_series.append((self.rel(), pct))

    def finger(self, count):
        self.finger_series.append((self.rel(), count))

    def color_trigger(self, color):
        self.color_counts[color] += 1
        self.color_sequence.append(color)

    def gesture_trigger(self, action):
        self.gesture_counts[action] += 1
        self.gesture_sequence.append(action)


# ===========================================================================
# Color detection / reaction
# ===========================================================================
def detect_color(hsv, lower, upper):
    """Return (bounding_box, area) for the largest blob, or None."""
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < MIN_AREA:
        return None
    return cv2.boundingRect(c), area


def react_to_color(tello, color, log):
    """Execute the straight move mapped to a color, then return to hover."""
    label = {"yellow": "move right", "blue": "move left", "purple": "move up"}[color]
    log.event("color", "maneuver", f"{color} -> {label} {COLOR_MOVE_CM}cm")
    try:
        if color == "yellow":
            tello.move_right(COLOR_MOVE_CM)
        elif color == "blue":
            tello.move_left(COLOR_MOVE_CM)
        elif color == "purple":
            tello.move_up(COLOR_MOVE_CM)
    except Exception as e:
        log.event("color", "maneuver_fail", f"{color}: {e}")
    time.sleep(0.5)
    log.color_trigger(color)


def do_gesture_move(tello, action, dist, fingers, log):
    """Execute a movement mapped to a finger count, then return to hover."""
    label = {"front": "move forward", "back": "move back"}[action]
    log.event("gesture", "maneuver", f"{fingers} fingers -> {label} {dist}cm")
    try:
        if action == "front":
            tello.move_forward(dist)
        elif action == "back":
            tello.move_back(dist)
    except Exception as e:
        log.event("gesture", "maneuver_fail", f"{action}: {e}")
    time.sleep(0.3)
    log.gesture_trigger(action)


# ===========================================================================
# Finger counting (angle based)
# ===========================================================================
def joint_angle_deg(lms, a, b, c):
    pa = np.array([lms[a].x, lms[a].y, lms[a].z])
    pb = np.array([lms[b].x, lms[b].y, lms[b].z])
    pc = np.array([lms[c].x, lms[c].y, lms[c].z])
    ba, bc = pa - pb, pc - pb
    nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
    if nba < 1e-6 or nbc < 1e-6:
        return 180.0
    cosang = np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)
    return math.degrees(math.acos(cosang))


def finger_states(lms):
    out = []
    for name, (a, b, c) in FINGER_JOINTS.items():
        ang = joint_angle_deg(lms, a, b, c)
        out.append((name, ang, ang >= EXTENDED_THRESHOLD_DEG[name]))
    return out


def count_fingers(lms):
    return sum(1 for _, _, ext in finger_states(lms) if ext)


def draw_hand(frame, lms, states=None, debug=False):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lms]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 200), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)
    if debug and states:
        for (name, ang, ext), (_, jb, tip) in zip(states, FINGER_JOINTS.values()):
            col = (0, 255, 0) if ext else (0, 0, 255)
            cv2.putText(frame, f"{int(ang)}", (int(lms[jb].x*w)+6, int(lms[jb].y*h)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            cv2.circle(frame, (int(lms[tip].x*w), int(lms[tip].y*h)), 8, col, -1)


# ===========================================================================
# HUD
# ===========================================================================
def draw_hud(frame, phase, color_triggers, finger_count, battery, fps, gesture_moves=0):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 32), (0, 0, 0), -1)
    cv2.putText(frame, f"PHASE: {phase.upper()}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    if phase == "color":
        msg = f"colors {color_triggers}/{MIN_COLOR_TRIGGERS}"
    else:
        fc = "--" if finger_count is None else str(finger_count)
        msg = f"fingers {fc} (2/3=front 5=back)  moves {gesture_moves}/{MIN_GESTURE_MOVES}"
    cv2.putText(frame, msg, (190, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"BAT:{battery}%  FPS:{fps:4.1f}", (w - 200, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# ===========================================================================
# Report generation  (runs at shutdown; pure matplotlib + csv)
# ===========================================================================
def generate_report(log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(REPORT_ROOT, stamp)
    os.makedirs(outdir, exist_ok=True)

    # ---- flight_log.csv ---------------------------------------------------
    with open(os.path.join(outdir, "flight_log.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["t", "phase", "event", "detail", "battery"])
        wr.writeheader()
        wr.writerows(log.events)

    duration = log.events[-1]["t"] if log.events else 0.0
    start_bat = log.battery_series[0][1] if log.battery_series else None
    end_bat = log.end_battery if log.end_battery is not None else (
        log.battery_series[-1][1] if log.battery_series else None)

    # ---- summary.csv ------------------------------------------------------
    summary = {
        "duration_s": duration,
        "color_triggers_total": sum(log.color_counts.values()),
        "yellow": log.color_counts["yellow"],
        "blue": log.color_counts["blue"],
        "purple": log.color_counts["purple"],
        "color_sequence": " -> ".join(log.color_sequence) or "none",
        "gesture_moves_total": sum(log.gesture_counts.values()),
        "move_front (2)": log.gesture_counts["front"],
        "move_back (5)": log.gesture_counts["back"],
        "gesture_sequence": " -> ".join(log.gesture_sequence) or "none",
        "landed_after_3_gestures": log.landed_on_gesture,
        "start_battery": start_bat,
        "end_battery": end_bat,
    }
    with open(os.path.join(outdir, "summary.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        for k, v in summary.items():
            wr.writerow([k, v])

    # ---- battery_vs_time.png ---------------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    if log.battery_series:
        ts, bs = zip(*log.battery_series)
        ax.plot(ts, bs, marker="o", color="#a14d4d", linewidth=2)
    ax.set_xlabel("Mission time (s)"); ax.set_ylabel("Battery (%)")
    ax.set_title("Battery level over the A to B run")
    ax.set_ylim(0, 100); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "battery_vs_time.png"), dpi=170)
    plt.close(fig)

    # ---- color_triggers.png ----------------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    cols = list(COLOR_ORDER)
    vals = [log.color_counts[c] for c in cols]
    ax.bar(cols, vals, color=["#c9b037", "#3b6ea5", "#7a6da8"], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.05, str(v), ha="center", fontsize=11, weight="bold")
    ax.set_ylabel("Maneuvers fired")
    ax.set_title(f"Color triggers (need >= {MIN_COLOR_TRIGGERS} to advance)")
    ax.set_ylim(0, max(vals + [MIN_COLOR_TRIGGERS]) + 1)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "color_triggers.png"), dpi=170)
    plt.close(fig)

    # ---- finger_timeline.png ---------------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    if log.finger_series:
        ts, fs = zip(*log.finger_series)
        ax.plot(ts, fs, color="#5b8c5a", linewidth=1.8)
    for cnt, txt in [(2, "front"), (3, "front"), (5, "back")]:
        ax.axhline(cnt, color="#c08a2e", linestyle="--", linewidth=0.9)
        ax.text(ax.get_xlim()[1], cnt, f" {cnt}={txt}", va="center",
                fontsize=8, color="#c08a2e")
    ax.set_xlabel("Mission time (s)"); ax.set_ylabel("Fingers detected")
    ax.set_title("Finger count during the gesture phase")
    ax.set_ylim(-0.2, 5.2); ax.set_yticks(range(6)); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "finger_timeline.png"), dpi=170)
    plt.close(fig)

    # ---- gesture_moves.png -----------------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    glabels = ["front (2)", "back (5)"]
    gvals = [log.gesture_counts["front"], log.gesture_counts["back"]]
    ax.bar(glabels, gvals, color=["#5b8c5a", "#3b6ea5"], width=0.5)
    for i, v in enumerate(gvals):
        ax.text(i, v + 0.05, str(v), ha="center", fontsize=11, weight="bold")
    ax.set_ylabel("Commands executed")
    ax.set_title("Gesture commands in the control zone")
    ax.set_ylim(0, max(gvals + [1]) + 1)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "gesture_moves.png"), dpi=170)
    plt.close(fig)


    # ---- summary_table.png -----------------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 3.2)); ax.axis("off")
    rows = [[k.replace("_", " "), str(v)] for k, v in summary.items()]
    tbl = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                   cellLoc="left", colLoc="left", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2E5E8C"); cell.set_text_props(color="white", weight="bold")
        cell.set_edgecolor("#bbbbbb")
    ax.set_title("Mission summary", weight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "summary_table.png"), dpi=170)
    plt.close(fig)

    print(f"\nReport assets written to: {os.path.abspath(outdir)}")
    return outdir


# ===========================================================================
# Main mission
# ===========================================================================
def main():
    import json
    with open(COLORS_FILE) as f:
        colors = json.load(f)

    # Build the hand detector (VIDEO mode, single hand)
    base = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base, running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1, min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
    detector = mp_vision.HandLandmarker.create_from_options(options)

    log = MissionLog()

    tello = Tello()
    tello.connect()
    batt = tello.get_battery()
    log.event("init", "connect", f"battery {batt}%", batt)
    log.battery(batt)
    if batt < MIN_BATTERY_PCT:
        raise SystemExit(f"Battery {batt}% < {MIN_BATTERY_PCT}% — charge before "
                         f"flying; low battery causes video drops and command timeouts.")

    # Give commands more time to ACK on a busy/weak Wi-Fi link.
    tello.RESPONSE_TIMEOUT = RESPONSE_TIMEOUT_SEC

    # Lighten the video load BEFORE turning the stream on. Constant names vary
    # slightly between djitellopy versions, so this is best-effort.
    try:
        tello.set_video_resolution(Tello.RESOLUTION_480P)
        tello.set_video_fps(Tello.FPS_30)
        tello.set_video_bitrate(Tello.BITRATE_AUTO)
    except Exception as e:
        log.event("init", "video_opts_skip", str(e))

    tello.streamon()
    reader = tello.get_frame_read()

    # Wait for REAL frames instead of a blind sleep — this is what prevents the
    # "Do not have enough frames for decoding" crash in the decode thread.
    t_wait = time.time()
    while True:
        f = reader.frame
        if f is not None and getattr(f, "size", 0) > 0:
            break
        if time.time() - t_wait > STREAM_WAIT_SEC:
            raise SystemExit(f"No video after {STREAM_WAIT_SEC}s — check Wi-Fi "
                             f"signal/interference, distance, or battery.")
        time.sleep(0.1)
    log.event("init", "stream_ready", f"video up in {time.time()-t_wait:.1f}s")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    phase = "color"
    streak_color, streak_n = None, 0          # color confirmation
    gesture_streak = 0                         # gesture confirmation count
    gesture_streak_count = None                # which finger count is streaking
    last_gesture_t = 0.0                       # cooldown timer
    debug = True
    flying = False
    fps, prev = 0.0, time.time()
    last_batt_t, battery = 0.0, batt
    gesture_t0 = None

    try:
        tello.takeoff()
        flying = True
        log.event("color", "takeoff", "point A")
        start = time.time()

        while True:
            if time.time() - start > MAX_FLIGHT_SEC:
                log.event(phase, "timeout", "max flight time reached")
                break

            frame = reader.frame
            if frame is None:
                cv2.waitKey(5); continue

            frame = cv2.resize(frame, (640, 480))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)   # Tello streams RGB

            # periodic battery sample
            if time.time() - last_batt_t > BATTERY_SAMPLE_SEC:
                battery = tello.get_battery()
                log.battery(battery)
                last_batt_t = time.time()

            finger_count = None

            # ---------------- PHASE 1: COLOR -----------------------------
            if phase == "color":
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                best, best_area, best_box = None, 0, None
                for name in COLOR_ORDER:
                    r = detect_color(hsv, colors[name]["lower"], colors[name]["upper"])
                    if r and r[1] > best_area:
                        best_box, best_area, best = r[0], r[1], name

                if best_box is not None:
                    x, y, w, h = best_box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), DRAW[best], 3)
                    cv2.putText(frame, best, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, DRAW[best], 2)

                if best == streak_color and best is not None:
                    streak_n += 1
                else:
                    streak_color, streak_n = best, (1 if best else 0)

                if streak_n >= COLOR_CONFIRM_FRAMES:
                    react_to_color(tello, streak_color, log)
                    streak_color, streak_n = None, 0
                    done = sum(log.color_counts.values())
                    if done >= MIN_COLOR_TRIGGERS:
                        phase = "gesture"
                        gesture_t0 = time.time()
                        log.event("gesture", "phase_change",
                                  f"{done} colors cleared, entering gesture zone")

            # ---------------- PHASE 2: GESTURE ---------------------------
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms = int((time.time() - gesture_t0) * 1000)
                result = detector.detect_for_video(mp_img, ts_ms)

                if result.hand_landmarks:
                    lms = result.hand_landmarks[0]
                    states = finger_states(lms)
                    finger_count = sum(1 for _, _, e in states if e)
                    draw_hand(frame, lms, states, debug)
                else:
                    finger_count = None

                log.finger(finger_count if finger_count is not None else 0)

                # Debounce: a count must persist GESTURE_CONFIRM_FRAMES frames,
                # then fire once; the gesture must drop/change and the cooldown
                # must elapse before the same command can fire again.
                if finger_count == gesture_streak_count and finger_count is not None:
                    gesture_streak += 1
                else:
                    gesture_streak_count, gesture_streak = finger_count, 1

                ready = (gesture_streak == GESTURE_CONFIRM_FRAMES
                         and (time.time() - last_gesture_t) > GESTURE_COOLDOWN_SEC)
                if ready and finger_count in FINGER_ACTIONS:
                    action, dist = FINGER_ACTIONS[finger_count]
                    do_gesture_move(tello, action, dist, finger_count, log)
                    last_gesture_t = time.time()
                    if sum(log.gesture_counts.values()) >= MIN_GESTURE_MOVES:
                        log.landed_on_gesture = True
                        log.event("gesture", "land_trigger",
                                  f"{MIN_GESTURE_MOVES} gestures done -> landing")
                        break
                # counts not in FINGER_ACTIONS (0, 1, 4) are ignored

            # ---------------- HUD + display ------------------------------
            now = time.time()
            dt = now - prev
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            prev = now
            draw_hud(frame, phase, sum(log.color_counts.values()),
                     finger_count, battery, fps, sum(log.gesture_counts.values()))
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                log.event(phase, "abort", "operator quit")
                break
            elif key == ord('d'):
                debug = not debug

    except KeyboardInterrupt:
        log.event(phase, "abort", "KeyboardInterrupt")
    finally:
        if flying:
            try:
                tello.land()
                log.event("land", "landed", "point B")
            except Exception as e:
                log.event("land", "emergency", str(e))
                tello.emergency()
        try:
            log.end_battery = tello.get_battery()
        except Exception:
            pass
        detector.close()
        cv2.destroyAllWindows()
        try:
            tello.streamoff()
        except Exception:
            pass
        tello.end()

        # Always produce the report assets, even on an aborted run
        try:
            generate_report(log)
        except Exception as e:
            print(f"Report generation failed: {e}")


if __name__ == "__main__":
    main()