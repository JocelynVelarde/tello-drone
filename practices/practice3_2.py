from djitellopy import Tello
import cv2
import time
import os
import threading

def main():
    tello = Tello()
    out = None
    recording = False

    try:
        print("Conectando al dron...")
        tello.connect()

        battery = tello.get_battery()
        print(f"Batería: {battery}%")

        tello.set_video_resolution(Tello.RESOLUTION_720P)
        tello.set_video_fps(Tello.FPS_30)
        tello.set_video_bitrate(Tello.BITRATE_4MBPS)

        tello.streamon()
        frame_read = tello.get_frame_read()
        time.sleep(1)  

        print("Despegando...")
        tello.takeoff()
        time.sleep(1) 

        frame = frame_read.frame
        h, w = frame.shape[:2]
        print(f"Resolución del stream: {w}x{h}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(os.getcwd(), "tello_circle.mp4")
        out = cv2.VideoWriter(output_path, fourcc, 30, (w, h))
        print(f"Grabando video en: {output_path}")

        # Background recording thread
        recording = True
        def record_loop():
            while recording:
                f = frame_read.frame
                if f is not None:
                    out.write(f) 
                time.sleep(1 / 30)

        record_thread = threading.Thread(target=record_loop, daemon=True)
        record_thread.start()

        def fly(duration):
            time.sleep(duration)

        print("Subiendo...")
        tello.send_rc_control(0, 0, 30, 0)
        fly(2)
        tello.send_rc_control(0, 0, 0, 0)
        fly(0.5)

        print("Iniciando círculo...")
        tello.rotate_clockwise(360)
        fly(4)

        print("Bajando...")
        tello.send_rc_control(0, 0, -30, 0)
        fly(2)
        tello.send_rc_control(0, 0, 0, 0)
        fly(0.5)

        print("Aterrizando...")
        tello.land()

    except Exception as e:
        print("Error:", e)

    finally:
        print("Cerrando conexión y guardando video...")
        recording = False
        time.sleep(0.5)
        if out:
            out.release()
        cv2.destroyAllWindows()
        tello.streamoff()
        tello.end()
        print("Video guardado como tello_circle.mp4")


if __name__ == "__main__":
    main()