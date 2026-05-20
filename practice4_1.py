from djitellopy import Tello
import cv2
import numpy as np
import time
import os
import threading

def main():
    tello = Tello()
    out = None
    recording = False
    record_thread = None

    try:
        print("Conectando al dron...")
        tello.connect()
        print(f"Batería: {tello.get_battery()}%")

        tello.set_video_resolution(Tello.RESOLUTION_720P)
        tello.set_video_fps(Tello.FPS_30)
        tello.set_video_bitrate(Tello.BITRATE_4MBPS)

        tello.streamon()
        frame_read = tello.get_frame_read()
        time.sleep(1)

        # Working frame size (matches assignment: 480x360)
        FRAME_W, FRAME_H = 480, 360
        CENTER_X = FRAME_W // 2   # 240
        CENTER_Y = FRAME_H // 2   # 180

        # HSV range for green
        lower = np.array([50, 100, 100])
        upper = np.array([70, 255, 255])

        # PD controller gains (from assignment)
        Kp_yaw, Kd_yaw = 0.4, 0.2
        Kp_z,   Kd_z   = 0.3, 0.1

        prev_error_yaw = 0
        prev_error_z = 0

        # Dead-zone thresholds to avoid jitter when object is near center
        YAW_DEADZONE = 25   # pixels
        Z_DEADZONE   = 20   # pixels

        # Video writer for evidence
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(os.getcwd(), "visual_tracking.mp4")
        out = cv2.VideoWriter(output_path, fourcc, 20, (FRAME_W, FRAME_H))
        print(f"Grabando en: {output_path}")

        # Latest annotated frame, shared with recorder thread
        latest_frame = {"img": None}
        frame_lock = threading.Lock()

        def record_loop():
            while recording:
                with frame_lock:
                    f = latest_frame["img"]
                if f is not None:
                    out.write(f)
                time.sleep(1 / 20)

        # Takeoff and stabilize
        print("Despegando...")
        tello.takeoff()
        time.sleep(2)

        # Start recording thread after takeoff
        recording = True
        record_thread = threading.Thread(target=record_loop, daemon=True)
        record_thread.start()

        print("Tracking iniciado. Presiona 'q' para aterrizar.")
        while True:
            frame = frame_read.frame
            if frame is None:
                continue

            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

            detected = False
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)

                # Ignore tiny noise blobs
                if w * h > 500:
                    detected = True
                    cx = x + w // 2
                    cy = y + h // 2

                    # --- Yaw control ---
                    error_yaw = cx - CENTER_X
                    if abs(error_yaw) < YAW_DEADZONE:
                        error_yaw = 0
                    derivative_yaw = error_yaw - prev_error_yaw
                    yaw_speed = int(Kp_yaw * error_yaw +
                                    Kd_yaw * derivative_yaw)
                    yaw_speed = int(np.clip(yaw_speed, -90, 90))

                    # --- Height (z) control ---
                    error_z = CENTER_Y - cy   # positive => object above center
                    if abs(error_z) < Z_DEADZONE:
                        error_z = 0
                    derivative_z = error_z - prev_error_z
                    ud = int(Kp_z * error_z + Kd_z * derivative_z)
                    ud = int(np.clip(ud, -20, 20))

                    # Send RC command: (left/right, fwd/back, up/down, yaw)
                    tello.send_rc_control(0, 0, ud, yaw_speed)

                    prev_error_yaw = error_yaw
                    prev_error_z = error_z

                    # Draw overlays
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                    cv2.putText(frame, "Green detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 2)
                    cv2.putText(frame,
                                f"yaw={yaw_speed} ud={ud}",
                                (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 255), 1)

            if not detected:
                # No object visible: hold position
                tello.send_rc_control(0, 0, 0, 0)
                prev_error_yaw = 0
                prev_error_z = 0
                cv2.putText(frame, "Searching...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)

            # Draw frame center crosshair
            cv2.drawMarker(frame, (CENTER_X, CENTER_Y), (255, 255, 255),
                           cv2.MARKER_CROSS, 20, 1)

            with frame_lock:
                latest_frame["img"] = frame.copy()

            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupción manual - aterrizando...")
    except Exception as e:
        print("Error:", e)
    finally:
        print("Cerrando recursos...")
        try:
            tello.send_rc_control(0, 0, 0, 0)
            time.sleep(0.3)
            tello.land()
        except Exception as e:
            print("Error al aterrizar:", e)

        recording = False
        time.sleep(0.5)
        if out:
            out.release()
        cv2.destroyAllWindows()
        try:
            tello.streamoff()
        except Exception:
            pass
        tello.end()
        print("Video guardado.")

if __name__ == "__main__":
    main()