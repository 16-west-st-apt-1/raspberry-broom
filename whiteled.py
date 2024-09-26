#!/usr/bin/env python3

import RPi.GPIO as GPIO
import time

# GPIO pin setup
RED_PIN = 27
GREEN_PIN =23
BLUE_PIN = 22

# PWM Frequency
FREQUENCY = 10000

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(RED_PIN, GPIO.OUT)
GPIO.setup(GREEN_PIN, GPIO.OUT)
GPIO.setup(BLUE_PIN, GPIO.OUT)

# Setup PWM
red_pwm = GPIO.PWM(RED_PIN, FREQUENCY)
green_pwm = GPIO.PWM(GREEN_PIN, FREQUENCY)
blue_pwm = GPIO.PWM(BLUE_PIN, FREQUENCY)

# Start PWM with 0 duty cycle (off)
red_pwm.start(0)
green_pwm.start(0)
blue_pwm.start(0)

def set_color(red, green, blue):
    """Set color of the RGB LED.
    Parameters:
        red (int): Red component (0-100)
        green (int): Green component (0-100)
        blue (int): Blue component (0-100)
    """
    red_pwm.ChangeDutyCycle(red)
    green_pwm.ChangeDutyCycle(green)
    blue_pwm.ChangeDutyCycle(blue)

try:
    while True:
        # Red
        set_color(25, 100, 25)
except KeyboardInterrupt:
    red_pwm.stop()
    green_pwm.stop()
    blue_pwm.stop()
    GPIO.cleanup()

