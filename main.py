#!/usr/bin/env python3

from enum import Enum
import cv2
from PiDust import PiDust
from picamera2 import Picamera2, Preview
import RPi.GPIO as GPIO
import time

class Motors(Enum):
    # GPIO output pin numbers
    RIGHT_A = 3 # right motor
    RIGHT_B = 2
    LEFT_A = 17 # left motor
    LEFT_B = 4

def initializeGPIO():
    # Use BCM (Broadcom SOC channel) numbering
    GPIO.setmode(GPIO.BCM)

    for motor in Motors:
        GPIO.setup(motor.value, GPIO.OUT)

    frequency = 1000

    motorLeftA = GPIO.PWM(Motors.LEFT_A.value, frequency)
    motorRightA = GPIO.PWM(Motors.RIGHT_A.value, frequency)
    return {"leftA": motorLeftA, "rightA": motorRightA}

def loadPiCam2():
    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration()
    picam2.configure(camera_config)
    #picam2.start_preview(Preview.DRM) # Only works with monitor
    picam2.start_preview(Preview.NULL)
    picam2.start()
    return picam2
    #time.sleep(2)
    #picam2.capture_file("test2.jpg")

def camLiveFeed(cam, fps):
    msDelay = int(1000 / fps)
    while True:
        image = cam.capture_array()
        cv2.imshow("Image", image)
        cv2.waitKey(msDelay) & 0xFF == ord("q")


def main():
    try:
        camera = loadPiCam2()
        motors = initializeGPIO()
        pidust = PiDust(camera, motors, 3)
        pidust.run()

        # Test video feed
        # cv2.namedWindow("Image", cv2.WND_PROP_TOPMOST)  # Resizable window
        # camLiveFeed(camera, 10)
    except KeyboardInterrupt:
        camera.stop()
        GPIO.cleanup()  # Clean up GPIO settings


if __name__ == "__main__":
    main()
