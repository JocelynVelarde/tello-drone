"""
Tello Drone Control Dashboard
Covers all commands from Activities 1, 2, and 3:
  - Basic connect / battery / takeoff / land
  - Absolute movement commands (move_forward, move_up, rotate_clockwise, etc.)
  - RC control (send_rc_control) for speed-based movement
  - Video streaming with resolution / fps / bitrate settings
  - Pre-built maneuvers: triangle, circle, spiral
"""

import streamlit as st
import threading
import time
import cv2
import os
import numpy as np
from datetime import datetime

# ─── Module-level globals (safe to access from background threads) ──────────
# st.session_state is NOT accessible from threads — store the drone object here.
_tello_instance   = None   # djitellopy Tello object
_frame_read       = None   # Tello frame reader
_video_writer     = None   # cv2.VideoWriter
_stop_record_flag = False  # signals recording thread to stop
_recording_active = False

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tello Flight Control",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

:root {
    --bg: #0a0e17;
    --panel: #111827;
    --border: #1e2d40;
    --accent: #00e5ff;
    --accent2: #ff6b35;
    --accent3: #39ff14;
    --text: #c8d8e8;
    --muted: #4a6080;
    --danger: #ff3b3b;
    --warn: #ffb703;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'Rajdhani', sans-serif;
    color: var(--text);
}

[data-testid="stSidebar"] {
    background: #0d1320 !important;
    border-right: 1px solid var(--border);
}

h1, h2, h3 {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.block-container { padding-top: 1.5rem; }

/* Cards */
.card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 18px 20px;
    margin-bottom: 14px;
    position: relative;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--accent);
    border-radius: 8px 0 0 8px;
}
.card.orange::before { background: var(--accent2); }
.card.green::before  { background: var(--accent3); }
.card.red::before    { background: var(--danger); }
.card.yellow::before { background: var(--warn); }

/* Status badges */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.badge-connected    { background: #003a1f; color: var(--accent3); border: 1px solid var(--accent3); }
.badge-disconnected { background: #2a0a0a; color: var(--danger);  border: 1px solid var(--danger); }
.badge-flying       { background: #003340; color: var(--accent);  border: 1px solid var(--accent); }
.badge-idle         { background: #1a1a00; color: var(--warn);    border: 1px solid var(--warn); }

/* Mono values */
.mono { font-family: 'Share Tech Mono', monospace; color: var(--accent); font-size: 1.1rem; }

/* Section headers */
.section-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 8px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 4px;
}

/* Metric boxes */
.metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
.metric-box {
    background: #0d1a2a;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 14px;
    text-align: center;
}
.metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.metric-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.3rem;
    color: var(--accent);
    font-weight: 700;
}

/* Log box */
.log-box {
    background: #070b12;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #7aa8c0;
    height: 180px;
    overflow-y: auto;
    white-space: pre-wrap;
}

/* Buttons */
.stButton > button {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    transition: all 0.15s !important;
}

/* Sliders */
.stSlider label { font-family: 'Share Tech Mono', monospace; font-size: 0.75rem; color: var(--muted); }

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-size: 0.85rem;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Session state defaults ─────────────────────────────────────────────────
# Only plain / serialisable values live in session_state.
# The Tello object, frame reader, and video writer are module-level globals
# so background threads can reach them without touching session_state.
def init_state():
    defaults = {
        "connected":   False,
        "flying":      False,
        "streaming":   False,
        "recording":   False,
        "log":         [],
        "battery":     0,
        "height":      0,
        "temp":        0,
        "flight_time": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── Helper: log ───────────────────────────────────────────────────────────
# log() only appends a string — safe from any thread because CPython's GIL
# makes list.insert() on a plain list atomic enough for our purposes.
def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    icons = {"INFO": "◆", "OK": "✔", "ERR": "✘", "WARN": "▲", "CMD": "▶"}
    icon = icons.get(level, "◆")
    entry = f"[{ts}] {icon} {msg}"
    # Safely append even from background threads
    st.session_state.log.insert(0, entry)
    if len(st.session_state.log) > 60:
        st.session_state.log = st.session_state.log[:60]


# ─── Tello accessor (uses module global, NOT session_state) ────────────────
def get_tello():
    return _tello_instance


# ─── Connection ────────────────────────────────────────────────────────────
def do_connect(ip):
    global _tello_instance
    try:
        from djitellopy import Tello
        t = Tello(host=ip)
        t.connect()
        _tello_instance = t
        st.session_state.connected = True
        st.session_state.battery   = t.get_battery()
        log(f"Connected to {ip} | Battery: {st.session_state.battery}%", "OK")
    except Exception as e:
        log(f"Connection failed: {e}", "ERR")

def do_disconnect():
    global _tello_instance, _frame_read
    t = _tello_instance
    if t:
        try:
            if st.session_state.streaming:
                t.streamoff()
            t.end()
        except:
            pass
    _tello_instance = None
    _frame_read     = None
    st.session_state.connected = False
    st.session_state.flying    = False
    st.session_state.streaming = False
    log("Disconnected", "WARN")


# ─── Basic flight ──────────────────────────────────────────────────────────
def do_takeoff():
    t = get_tello()
    if not t: return
    try:
        t.takeoff()
        st.session_state.flying = True
        log("Takeoff", "CMD")
    except Exception as e:
        log(f"Takeoff error: {e}", "ERR")

def do_land():
    t = get_tello()
    if not t: return
    try:
        t.land()
        st.session_state.flying = False
        log("Land", "CMD")
    except Exception as e:
        log(f"Land error: {e}", "ERR")

def do_emergency():
    t = get_tello()
    if not t: return
    try:
        t.emergency()
        st.session_state.flying = False
        log("EMERGENCY STOP", "ERR")
    except Exception as e:
        log(f"Emergency error: {e}", "ERR")

def refresh_telemetry():
    t = get_tello()
    if not t or not st.session_state.connected: return
    try:
        st.session_state.battery     = t.get_battery()
        st.session_state.height      = t.get_height()
        st.session_state.temp        = t.get_temperature()
        st.session_state.flight_time = t.get_flight_time()
        log("Telemetry refreshed", "OK")
    except Exception as e:
        log(f"Telemetry error: {e}", "ERR")


# ── Activity 1: Absolute movement ──────────────────────────────────────────
def do_absolute_move(direction, distance):
    t = get_tello()          # reads module global — safe from threads
    if not t: return
    try:
        cmd_map = {
            "Forward": t.move_forward,
            "Back":    t.move_back,
            "Left":    t.move_left,
            "Right":   t.move_right,
            "Up":      t.move_up,
            "Down":    t.move_down,
        }
        cmd_map[direction](distance)
        log(f"Move {direction} {distance}cm", "CMD")
    except Exception as e:
        log(f"Move error: {e}", "ERR")

def do_rotate(direction, degrees):
    t = get_tello()
    if not t: return
    try:
        if direction == "Clockwise":
            t.rotate_clockwise(degrees)
        else:
            t.rotate_counter_clockwise(degrees)
        log(f"Rotate {direction} {degrees}°", "CMD")
    except Exception as e:
        log(f"Rotate error: {e}", "ERR")

def do_flip(direction):
    t = get_tello()
    if not t: return
    try:
        flip_map = {"Left": "l", "Right": "r", "Forward": "f", "Back": "b"}
        t.flip(flip_map[direction])
        log(f"Flip {direction}", "CMD")
    except Exception as e:
        log(f"Flip error: {e}", "ERR")


# ── Activity 2: RC control ─────────────────────────────────────────────────
def do_rc(a, b, c, d, duration):
    t = get_tello()          # module global — accessible from thread
    if not t: return
    try:
        log(f"RC → roll={a} pitch={b} throttle={c} yaw={d} for {duration}s", "CMD")
        t.send_rc_control(a, b, c, d)
        time.sleep(duration)
        t.send_rc_control(0, 0, 0, 0)
        time.sleep(0.4)
        log("RC done, stabilized", "OK")
    except Exception as e:
        log(f"RC error: {e}", "ERR")

def run_in_thread(fn, *args):
    th = threading.Thread(target=fn, args=args, daemon=True)
    th.start()


# ── Pre-built maneuvers ────────────────────────────────────────────────────
def maneuver_triangle():
    t = get_tello()
    if not t: return
    def _fly():
        log("Maneuver: Triangle start", "CMD")
        for i in range(3):
            t.send_rc_control(0, 30, 0, 0);  time.sleep(2.5)
            t.send_rc_control(0, 0,  0, 0);  time.sleep(0.5)
            t.send_rc_control(0, 0,  0, 30); time.sleep(2.0)
            t.send_rc_control(0, 0,  0, 0);  time.sleep(0.5)
            log(f"  Triangle leg {i+1}/3 done", "OK")
        log("Maneuver: Triangle complete", "OK")
    run_in_thread(_fly)

def maneuver_circle(fwd_speed, yaw_speed, duration):
    t = get_tello()
    if not t: return
    def _fly():
        log(f"Maneuver: Circle (fwd={fwd_speed}, yaw={yaw_speed}, {duration}s)", "CMD")
        t.send_rc_control(0, fwd_speed, 0, yaw_speed)
        time.sleep(duration)
        t.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)
        log("Maneuver: Circle complete", "OK")
    run_in_thread(_fly)

def maneuver_spiral(fwd, throttle, yaw, duration):
    t = get_tello()
    if not t: return
    def _fly():
        log(f"Maneuver: Spiral (fwd={fwd}, up={throttle}, yaw={yaw}, {duration}s)", "CMD")
        t.send_rc_control(0, fwd, throttle, yaw)
        time.sleep(duration)
        t.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)
        log("Maneuver: Spiral complete", "OK")
    run_in_thread(_fly)


# ── Activity 3: Video ──────────────────────────────────────────────────────
def do_stream_on(res, fps_val, bitrate):
    global _frame_read
    t = get_tello()
    if not t: return
    try:
        from djitellopy import Tello as _T
        res_map   = {"720p": _T.RESOLUTION_720P, "480p": _T.RESOLUTION_480P}
        fps_map   = {30: _T.FPS_30, 15: _T.FPS_15}
        brate_map = {4: _T.BITRATE_4MBPS, 2: _T.BITRATE_2MBPS, 1: _T.BITRATE_1MBPS}
        t.set_video_resolution(res_map.get(res, _T.RESOLUTION_720P))
        t.set_video_fps(fps_map.get(fps_val, _T.FPS_30))
        t.set_video_bitrate(brate_map.get(bitrate, _T.BITRATE_4MBPS))
        t.streamon()
        _frame_read = t.get_frame_read()       # store in module global
        st.session_state.streaming = True
        time.sleep(1.5)
        log(f"Stream ON | {res} @ {fps_val}fps | {bitrate}Mbps", "OK")
    except Exception as e:
        log(f"Stream error: {e}", "ERR")

def do_stream_off():
    global _frame_read
    t = get_tello()
    if not t: return
    try:
        t.streamoff()
        _frame_read = None
        st.session_state.streaming = False
        log("Stream OFF", "WARN")
    except Exception as e:
        log(f"Stream off error: {e}", "ERR")

def do_start_recording(output_path):
    global _video_writer, _stop_record_flag, _recording_active
    if not st.session_state.streaming or _frame_read is None:
        log("Cannot record: stream not active", "WARN")
        return
    frame = _frame_read.frame
    if frame is None:
        log("Cannot record: no frame yet", "WARN")
        return
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    _video_writer      = cv2.VideoWriter(output_path, fourcc, 30, (w, h))
    _stop_record_flag  = False
    _recording_active  = True
    st.session_state.recording = True

    def _record():
        global _recording_active
        while not _stop_record_flag:
            f = _frame_read.frame   # module global — safe from thread
            if f is not None:
                _video_writer.write(f)
            time.sleep(1 / 30)
        _video_writer.release()
        _recording_active = False
        log(f"Recording saved: {output_path}", "OK")

    threading.Thread(target=_record, daemon=True).start()
    log(f"Recording started → {output_path}", "CMD")

def do_stop_recording():
    global _stop_record_flag
    _stop_record_flag = True
    st.session_state.recording = False
    log("Recording stopped", "WARN")


# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚁 TELLO FC")
    st.markdown('<div class="section-title">Connection</div>', unsafe_allow_html=True)

    drone_ip = st.text_input("Drone IP", value="192.168.10.1", label_visibility="collapsed")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Connect", use_container_width=True, type="primary",
                     disabled=st.session_state.connected):
            do_connect(drone_ip)
    with c2:
        if st.button("Disconnect", use_container_width=True,
                     disabled=not st.session_state.connected):
            do_disconnect()

    st.markdown("---")

    # Status
    conn_badge = '<span class="badge badge-connected">CONNECTED</span>' if st.session_state.connected \
                 else '<span class="badge badge-disconnected">OFFLINE</span>'
    fly_badge  = '<span class="badge badge-flying">AIRBORNE</span>' if st.session_state.flying \
                 else '<span class="badge badge-idle">GROUNDED</span>'
    st.markdown(f"{conn_badge} &nbsp; {fly_badge}", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Telemetry</div>', unsafe_allow_html=True)

    if st.button("⟳ Refresh", use_container_width=True, disabled=not st.session_state.connected):
        refresh_telemetry()

    bat = st.session_state.battery
    bat_color = "#39ff14" if bat > 50 else ("#ffb703" if bat > 20 else "#ff3b3b")
    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-box">
        <div class="metric-label">Battery</div>
        <div class="metric-value" style="color:{bat_color}">{bat}%</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Height</div>
        <div class="metric-value">{st.session_state.height}cm</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Temp</div>
        <div class="metric-value">{st.session_state.temp}°C</div>
      </div>
    </div>
    <div style="margin-top:8px" class="metric-box">
      <div class="metric-label">Flight Time</div>
      <div class="metric-value">{st.session_state.flight_time}s</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Flight log</div>', unsafe_allow_html=True)
    log_text = "\n".join(st.session_state.log) if st.session_state.log else "Awaiting commands..."
    st.markdown(f'<div class="log-box">{log_text}</div>', unsafe_allow_html=True)
    if st.button("Clear Log", use_container_width=True):
        st.session_state.log = []

# ── Main header ─────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="font-size:2rem; margin-bottom:0; letter-spacing:0.15em; color:#00e5ff;">
  ◈ TELLO FLIGHT CONTROL DASHBOARD
</h1>
<p style="font-family:'Share Tech Mono',monospace; color:#4a6080; font-size:0.75rem; margin-top:2px; letter-spacing:0.1em;">
  ACTIVITIES 1 · 2 · 3 — ABSOLUTE MOVEMENT · RC CONTROL · VIDEO STREAM
</p>
<hr style="border-color:#1e2d40; margin: 10px 0 20px;">
""", unsafe_allow_html=True)

# ── Emergency always visible ────────────────────────────────────────────────
col_em, col_tf, col_ld, col_hov = st.columns(4)
with col_em:
    if st.button("🛑 EMERGENCY STOP", use_container_width=True, type="primary"):
        do_emergency()
with col_tf:
    if st.button("🚀 Takeoff", use_container_width=True,
                 disabled=not st.session_state.connected or st.session_state.flying):
        do_takeoff()
with col_ld:
    if st.button("🛬 Land", use_container_width=True,
                 disabled=not st.session_state.flying):
        do_land()
with col_hov:
    if st.button("⏸ Hover (RC stop)", use_container_width=True,
                 disabled=not st.session_state.flying):
        t = get_tello()
        if t:
            t.send_rc_control(0, 0, 0, 0)
            log("Hover / RC zeroed", "CMD")

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📐 Activity 1 · Absolute",
    "🕹️ Activity 2 · RC Control",
    "🎬 Activity 3 · Video",
    "✈️ Maneuvers",
])


# ══════════════════════════════════════════════════════════
# TAB 1 — Absolute movement
# ══════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Absolute Movement Commands</div>', unsafe_allow_html=True)
    st.caption("These commands move the drone a precise distance in cm (20–500). Drone must be airborne.")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Directional Move**")
        direction = st.selectbox("Direction", ["Forward", "Back", "Left", "Right", "Up", "Down"])
        distance  = st.slider("Distance (cm)", 20, 500, 50, 10)
        if st.button(f"▶ Move {direction} {distance}cm", use_container_width=True,
                     disabled=not st.session_state.flying):
            run_in_thread(do_absolute_move, direction, distance)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="card orange">', unsafe_allow_html=True)
        st.markdown("**Rotation**")
        rot_dir = st.selectbox("Rotation Direction", ["Clockwise", "Counter-Clockwise"])
        degrees = st.slider("Degrees", 1, 360, 90, 5)
        if st.button(f"▶ Rotate {rot_dir} {degrees}°", use_container_width=True,
                     disabled=not st.session_state.flying):
            run_in_thread(do_rotate, rot_dir, degrees)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card green">', unsafe_allow_html=True)
    st.markdown("**Flip** (requires sufficient battery ≥ 50%)")
    flip_cols = st.columns(4)
    for idx, fd in enumerate(["Left", "Right", "Forward", "Back"]):
        with flip_cols[idx]:
            if st.button(f"↻ Flip {fd}", use_container_width=True,
                         disabled=not st.session_state.flying):
                run_in_thread(do_flip, fd)
    st.markdown('</div>', unsafe_allow_html=True)

    # Visual pad
    st.markdown('<div class="section-title" style="margin-top:16px">Quick D-Pad</div>', unsafe_allow_html=True)
    pad_dist = st.slider("Quick Move Distance (cm)", 20, 200, 50, 10, key="pad_dist")
    _, pc, _ = st.columns([1, 1, 1])
    with pc:
        if st.button("▲ Forward", use_container_width=True, disabled=not st.session_state.flying):
            run_in_thread(do_absolute_move, "Forward", pad_dist)
    pl, pm, pr = st.columns(3)
    with pl:
        if st.button("◄ Left", use_container_width=True, disabled=not st.session_state.flying):
            run_in_thread(do_absolute_move, "Left", pad_dist)
    with pm:
        if st.button("▼ Back", use_container_width=True, disabled=not st.session_state.flying):
            run_in_thread(do_absolute_move, "Back", pad_dist)
    with pr:
        if st.button("► Right", use_container_width=True, disabled=not st.session_state.flying):
            run_in_thread(do_absolute_move, "Right", pad_dist)
    pu_col, pd_col = st.columns(2)
    with pu_col:
        if st.button("↑ Up", use_container_width=True, disabled=not st.session_state.flying):
            run_in_thread(do_absolute_move, "Up", pad_dist)
    with pd_col:
        if st.button("↓ Down", use_container_width=True, disabled=not st.session_state.flying):
            run_in_thread(do_absolute_move, "Down", pad_dist)


# ══════════════════════════════════════════════════════════
# TAB 2 — RC Control
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">RC Speed Control (send_rc_control)</div>', unsafe_allow_html=True)
    st.caption("Controls speed on each axis simultaneously. Values: -100 to 100. Duration sets how long the command runs.")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    rc_col1, rc_col2 = st.columns(2)
    with rc_col1:
        rc_a = st.slider("a — Roll   (left/right)", -100, 100, 0, key="rc_a")
        rc_b = st.slider("b — Pitch  (forward/back)", -100, 100, 0, key="rc_b")
    with rc_col2:
        rc_c = st.slider("c — Throttle (up/down)", -100, 100, 0, key="rc_c")
        rc_d = st.slider("d — Yaw    (rotation)", -100, 100, 0, key="rc_d")
    rc_dur = st.slider("Duration (seconds)", 0.5, 10.0, 2.0, 0.5, key="rc_dur")

    st.markdown(f"""
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.8rem; color:#4a6080; margin:8px 0;">
    send_rc_control(a=<span style="color:#00e5ff">{rc_a}</span>,
    b=<span style="color:#ff6b35">{rc_b}</span>,
    c=<span style="color:#39ff14">{rc_c}</span>,
    d=<span style="color:#ffb703">{rc_d}</span>) for {rc_dur}s
    </div>""", unsafe_allow_html=True)

    if st.button("▶ Send RC Command", use_container_width=True,
                 disabled=not st.session_state.flying, type="primary"):
        run_in_thread(do_rc, rc_a, rc_b, rc_c, rc_d, rc_dur)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:14px">RC Quick Presets</div>', unsafe_allow_html=True)
    presets = {
        "→ Strafe Right":    (30, 0, 0, 0),
        "← Strafe Left":     (-30, 0, 0, 0),
        "↑ Climb":           (0, 0, 30, 0),
        "↓ Descend":         (0, 0, -30, 0),
        "⟳ Yaw Right":       (0, 0, 0, 30),
        "⟲ Yaw Left":        (0, 0, 0, -30),
        "⬆ Pitch Forward":   (0, 30, 0, 0),
        "⬇ Pitch Back":      (0, -30, 0, 0),
    }
    preset_dur = st.slider("Preset Duration (s)", 0.5, 5.0, 1.5, 0.5, key="preset_dur")
    p_cols = st.columns(4)
    for i, (label, (a, b, c, d)) in enumerate(presets.items()):
        with p_cols[i % 4]:
            if st.button(label, use_container_width=True, disabled=not st.session_state.flying, key=f"preset_{i}"):
                run_in_thread(do_rc, a, b, c, d, preset_dur)


# ══════════════════════════════════════════════════════════
# TAB 3 — Video
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Video Stream Settings</div>', unsafe_allow_html=True)

    v_col1, v_col2, v_col3 = st.columns(3)
    with v_col1:
        res_choice = st.selectbox("Resolution", ["720p", "480p"])
    with v_col2:
        fps_choice = st.selectbox("FPS", [30, 15])
    with v_col3:
        brate_choice = st.selectbox("Bitrate (Mbps)", [4, 2, 1])

    s1, s2 = st.columns(2)
    with s1:
        if st.button("📡 Stream ON", use_container_width=True,
                     disabled=not st.session_state.connected or st.session_state.streaming):
            do_stream_on(res_choice, fps_choice, brate_choice)
    with s2:
        if st.button("📵 Stream OFF", use_container_width=True,
                     disabled=not st.session_state.streaming):
            do_stream_off()

    st.markdown('<div class="section-title" style="margin-top:14px">Recording</div>', unsafe_allow_html=True)
    rec_path = st.text_input("Output file path", value=os.path.join(os.getcwd(), "tello_recording.mp4"))
    r1, r2 = st.columns(2)
    with r1:
        if st.button("⏺ Start Recording", use_container_width=True,
                     disabled=not st.session_state.streaming or st.session_state.recording):
            do_start_recording(rec_path)
    with r2:
        if st.button("⏹ Stop Recording", use_container_width=True,
                     disabled=not st.session_state.recording):
            do_stop_recording()

    if st.session_state.recording:
        st.success("🔴 Recording in progress...")

    # Live frame viewer
    st.markdown('<div class="section-title" style="margin-top:14px">Live Frame Preview</div>', unsafe_allow_html=True)
    frame_placeholder = st.empty()
    if st.session_state.streaming and _frame_read is not None:
        if st.button("📸 Capture Frame", use_container_width=True):
            fr = _frame_read
            if fr and fr.frame is not None:
                frame_rgb = cv2.cvtColor(fr.frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, caption="Latest frame", use_container_width=True)
                log("Frame captured", "OK")
    else:
        frame_placeholder.markdown(
            '<div class="card" style="text-align:center;color:#4a6080;padding:40px;">'
            '📷 Stream not active</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# TAB 4 — Pre-built Maneuvers
# ══════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Pre-built Flight Maneuvers</div>', unsafe_allow_html=True)
    st.caption("These run in a background thread — you can still use other controls while they execute.")

    man_col1, man_col2, man_col3 = st.columns(3)

    with man_col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### △ Triangle")
        st.markdown("3 forward legs + 120° clockwise rotation × 3 using RC control.")
        if st.button("▶ Fly Triangle", use_container_width=True,
                     disabled=not st.session_state.flying, key="tri"):
            maneuver_triangle()
        st.markdown('</div>', unsafe_allow_html=True)

    with man_col2:
        st.markdown('<div class="card orange">', unsafe_allow_html=True)
        st.markdown("### ○ Circle")
        st.markdown("Forward + yaw simultaneously = circular arc.")
        c_fwd  = st.slider("Forward Speed", 10, 60, 30, key="c_fwd")
        c_yaw  = st.slider("Yaw Speed", 5, 50, 20, key="c_yaw")
        c_dur  = st.slider("Duration (s)", 5.0, 30.0, 18.0, 1.0, key="c_dur")
        st.caption(f"Radius ≈ larger when fwd/yaw ratio is high")
        if st.button("▶ Fly Circle", use_container_width=True,
                     disabled=not st.session_state.flying, key="circ"):
            maneuver_circle(c_fwd, c_yaw, c_dur)
        st.markdown('</div>', unsafe_allow_html=True)

    with man_col3:
        st.markdown('<div class="card green">', unsafe_allow_html=True)
        st.markdown("### ↗ Spiral")
        st.markdown("Circle + ascending throttle = helix path.")
        s_fwd  = st.slider("Forward Speed", 10, 50, 15, key="s_fwd")
        s_thr  = st.slider("Throttle (up)", 5, 40, 10, key="s_thr")
        s_yaw  = st.slider("Yaw Speed", 10, 50, 30, key="s_yaw")
        s_dur  = st.slider("Duration (s)", 3.0, 20.0, 6.0, 1.0, key="s_dur")
        if st.button("▶ Fly Spiral", use_container_width=True,
                     disabled=not st.session_state.flying, key="spir"):
            maneuver_spiral(s_fwd, s_thr, s_yaw, s_dur)
        st.markdown('</div>', unsafe_allow_html=True)

    # RC param reference
    st.markdown('<div class="section-title" style="margin-top:20px">RC Parameter Reference</div>', unsafe_allow_html=True)
    st.markdown("""
| Param | Axis | Positive | Negative |
|-------|------|----------|----------|
| `a` | Roll | Strafe right | Strafe left |
| `b` | Pitch | Move forward | Move back |
| `c` | Throttle | Ascend | Descend |
| `d` | Yaw | Rotate clockwise | Rotate counter-clockwise |
""")
    st.caption("Values range from -100 to 100. Always send (0,0,0,0) after a command to stabilize the drone.")