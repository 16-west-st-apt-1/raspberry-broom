#!/usr/bin/env python3

import RPi.GPIO as GPIO
import time

# Use BCM (Broadcom SOC channel) numbering
GPIO.setmode(GPIO.BCM)

# Set up the GPIO pin as an output
M1A = 2
M1B = 3

M2A = 4
M2B = 17

GPIO.setup(M1A, GPIO.OUT)
GPIO.setup(M1B, GPIO.OUT)
GPIO.setup(M2A, GPIO.OUT)
GPIO.setup(M2B, GPIO.OUT)

M1A_PWM =GPIO.PWM(2, 1000)
M2A_PWM = GPIO.PWM(4, 1000)

M1A_PWM.start(0)
M2A_PWM.start(0)

for dc in range(0, 101, 5):
    M1A_PWM.ChangeDutyCycle(dc)
    M2A_PWM.ChangeDutyCycle(dc)
    time.sleep(0.1)

M1A_PWM.ChangeDutyCycle(0)
M2A_PWM.ChangeDutyCycle(0)
# Blink the LED
#GPIO.output(M1A, GPIO.HIGH)  # Turn LED on   # Wait for 1 second
#GPIO.output(M1B, GPIO.LOW)
#GPIO.output(M2A, GPIO   # Turn LED off
time.sleep(5)
GPIO.output(M1A, GPIO.LOW)
GPIO.output(M2A, GPIO.LOW)                 # Wait for 1 second
GPIO.cleanup()  # Clean up GPIO settings
