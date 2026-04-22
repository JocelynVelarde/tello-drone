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

        print("Hover 1 segundos...")
        time.sleep(1)

        tello.move_up(50)

        for _ in range(3):
            tello.move_forward(80)  
            tello.rotate_clockwise(120)

        tello.move_down(50)

        print("Aterrizando...")
        tello.land()

    except Exception as e:
        print("Error:", e)

    finally:
        print("Cerrando conexión...")
        tello.end()


if __name__ == "__main__":
    main()