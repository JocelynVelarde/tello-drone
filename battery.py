from djitellopy import Tello
import streamlit as st
import time

def main():
    tello = Tello()

    try:
        print("Conectando al dron...")
        tello.connect()

        battery = tello.get_battery()
        print(f"Batería: {battery}%")

    finally:
        print("Cerrando conexión...")
        tello.end()

if __name__ == "__main__":
    main()