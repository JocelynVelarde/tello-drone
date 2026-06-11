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

        print("Hover 5 segundos...")
        time.sleep(5)

        print("Aterrizando...")
        tello.land()

    except Exception as e:
        print("Error:", e)

    finally:
        print("Cerrando conexión...")
        tello.end()


if __name__ == "__main__":
    main()