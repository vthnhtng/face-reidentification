import serial
import time

class UARTCommunicator:
    def __init__(self, port='COM4', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_enabled = False
        self.ser = None
        self.initialize_serial()

    def initialize_serial(self):
        """Initialize serial connection with ESP32"""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Wait for ESP32 to start up
            self.serial_enabled = True
            print(f"Successfully connected to ESP32 on {self.port}")
        except serial.SerialException as e:
            if "PermissionError" in str(e):
                print(f"Error: Cannot access {self.port} - Port may be in use by another program")
                print(f"Please close any other applications using {self.port} and try again")
            else:
                print(f"Warning: Could not open serial port: {e}")
            self.serial_enabled = False

    def send_open_signal_to_esp32(self):
        """Send signal to ESP32"""
        if self.serial_enabled and self.ser:
            try:
                self.ser.write(b'OPEN\n')
                print("Signal sent to ESP32")
            except serial.SerialException as e:
                print(f"Error sending signal: {e}")
                # If we lose connection, try to re-initialize
                self.initialize_serial()
        else:
            print("Serial communication is disabled - skipping signal")

    def close(self):
        """Close the serial connection"""
        if self.ser:
            self.ser.close()
