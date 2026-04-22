from djitellopy import Tello
import time

def main():
    tello = Tello()

    try:
        print("Conectando al dron...")
        tello.connect()

        battery = tello.get_battery()
        print(f"Batería: {battery}%")

        print("Despegando...")
        tello.takeoff()

        print("Hover 2 segundos...")
        time.sleep(1)

        # send_rc_control(left_right_velocity, forward_backward_velocity, up_down_velocity, yaw_velocity)

        # Move up to a safe height
        tello.send_rc_control(0, 0, 30, 0)
        time.sleep(1)
        tello.send_rc_control(0, 0, 0, 0)  
        time.sleep(0.5)

        print("Iniciando círculo...")
        # Forward (b) + clockwise yaw (d) at the same time = circle
        # Adjust speed and duration to control circle size
        tello.send_rc_control(0, 40, 0, 40)
        time.sleep(15)                      
        tello.send_rc_control(0, 0, 0, 0)   
        time.sleep(0.5)

        # Come back down
        tello.send_rc_control(0, 0, -30, 0)
        time.sleep(1)
        tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)

        print("Aterrizando...")
        tello.land()

    except Exception as e:
        print("Error:", e)

    finally:
        print("Cerrando conexión...")
        tello.end()


if __name__ == "__main__":
    main()