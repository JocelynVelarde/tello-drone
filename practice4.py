from djitellopy import Tello
import cv2
import numpy as np
import time
import os

def main():
    tello = Tello()
    out = None

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

        # Frame size we'll work with (matches assignment: 480x360)
        FRAME_W, FRAME_H = 480, 360

        # HSV range for green (from assignment)
        lower = np.array([50, 100, 100])
        upper = np.array([70, 255, 255])

        # Video writer for evidence
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(os.getcwd(), "color_detection.mp4")
        out = cv2.VideoWriter(output_path, fourcc, 20, (FRAME_W, FRAME_H))
        print(f"Grabando en: {output_path}")

        print("Detección iniciada. Presiona 'q' para salir.")
        while True:
            frame = frame_read.frame
            if frame is None:
                continue

            # Resize to standard size
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))

            # djitellopy returns RGB; convert to BGR for OpenCV consistency
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Build green mask
            mask = cv2.inRange(hsv, lower, upper)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)

                # Filter out tiny noise blobs
                if w * h > 300:
                    cx = x + w // 2
                    cy = y + h // 2

                    # Average color in the detected region (BGR)
                    object_region = frame[y:y+h, x:x+w]
                    avg_color = cv2.mean(object_region)[:3]
                    print(f"RGB color: R={avg_color[2]:.2f}, "
                          f"G={avg_color[1]:.2f}, B={avg_color[0]:.2f}")

                    # Draw bounding box and center
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                    cv2.putText(frame, "Green detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 2)
                else:
                    print("[INFO] Green not detected.")
            else:
                print("[INFO] Green not detected.")

            out.write(frame)
            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupción manual.")
    except Exception as e:
        print("Error:", e)
    finally:
        print("Cerrando recursos...")
        if out:
            out.release()
        cv2.destroyAllWindows()
        try:
            tello.streamoff()
        except Exception:
            pass
        tello.end()
        print("Listo.")

if __name__ == "__main__":
    main()