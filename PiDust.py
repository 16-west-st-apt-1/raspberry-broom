#!/usr/bin/env python3

import cv2
import numpy as np
from Path import Path
import RPi.GPIO as GPIO
import sys
import time
from Turn import Turn


class PiDust:
    def __init__(self, camera, motors):
        # Lines with this angle wrto horizontal are treated as horizontal lines.
        self.MAX_HORIZ_LINE_ANGLE = 20

        # Minimum length of lines found with HoughLinesP transform.
        self.MIN_LINE_LENGTH = 100

        # Horizontal lines should be within these margin percentages of the image.
        self.LEFT_MARGIN: float = 0.15
        self.RIGHT_MARGIN: float = 1 - self.LEFT_MARGIN

        # Baseline duty cycle for testing
        self.TEST_BASE_DC = 25
        self.baseSpeed = 25

        # Time between taking a new image
        self.PROCESS_TIME = 0.2

        # Load Path
        self.path = Path()

        # Store camera
        self.cam = camera

        # Store and start motors
        self.leftMotor = motors["leftA"]
        self.rightMotor = motors["rightA"]
        self.startMotors()

    def startMotors(self):
        self.leftMotor.start(0)
        self.rightMotor.start(0)

    def setLeftMotorDutyCycle(self, dc):
        self.leftMotor.ChangeDutyCycle(dc)

    def setRightMotorDutyCycle(self, dc):
        self.rightMotor.ChangeDutyCycle(dc)

    def stopMotors(self):
        self.setLeftMotorDutyCycle(0)
        self.setRightMotorDutyCycle(0)

    def go(self):
        self.setLeftMotorDutyCycle(self.TEST_BASE_DC)
        self.setRightMotorDutyCycle(self.TEST_BASE_DC)

    def leftTurn(self):
        """Turn left 90 degrees."""
        print("Turning left.")
        self.setLeftMotorDutyCycle(0)
        time.sleep(2)
        self.setLeftMotorDutyCycle(self.TEST_BASE_DC)

    def rightTurn(self):
        """Turn right 90 degrees."""
        print("Turning right.")
        self.setRightMotorDutyCycle(0)
        time.sleep(2)
        self.setRightMotorDutyCycle(self.TEST_BASE_DC)

    def turn(self):
        """Turn according to the turn type specified by Path."""
        turn = self.path.turnCode
        if turn == Turn.L:
            self.leftTurn()
        if turn == Turn.R:
            self.rightTurn()

    def getImage(self, filepath):
        """Get an image from the camera."""
        return cv2.imread(filepath)

    def preprocess(self, image):
        """Preprocess the image before OpenCV analysis."""

        # Grayscale
        gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        # Blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Threshold >100 -> black
        ret, thresh1 = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)

        # Erode to eliminate noise
        mask = cv2.erode(thresh1, None, iterations=2)
        # Dilate to restore eroded parts of image
        mask = cv2.dilate(mask, None, iterations=2)
        return mask

    def isHorizontal(self, x1, y1, x2, y2) -> bool:
        lineAngle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        return (
            np.abs(lineAngle) < self.MAX_HORIZ_LINE_ANGLE
            or np.abs(lineAngle - 180) < self.MAX_HORIZ_LINE_ANGLE
        )

    def isCloseToEdge(self, x1, x2, width) -> bool:
        """Is a line close enough to either vertical edge of the image?"""
        return x1 < width * self.LEFT_MARGIN or x2 > width * self.RIGHT_MARGIN

    def findVerticalThird(self, y1: int, height: int) -> int:
        """Return int representing which third of image height y-coord is in.

        0: top
        1: middle
        2: bottom
        """
        top = 0.15 * height
        middle = 0.75 * height

        if y1 < top:
            return 0
        elif y1 >= top and y1 <= middle:
            return 1
        else:
            return 2

    def findIntersection(self, image):
        """Detect if an image is at a corner.

        This requires some fine-tuning based on the distance between camera and the
        floor, the resolution, and the width and shapes of the lines being tracked."""
        # print("Checking for corners.")
        height, width = image.shape

        # Apply edge detection
        edges = cv2.Canny(image, 50, 150, apertureSize=3)

        # Perform Hough Line Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=self.MIN_LINE_LENGTH,
            maxLineGap=100,
        )

        if lines is not None:
            # Number of lines found
            nHoughLines = len(lines)
            # Number of lines representing intersection found
            nIntersectionLines = 0

            # Find lines
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if self.isHorizontal(x1, y1, x2, y2):
                        if self.isCloseToEdge(x1, x2, width):
                            nIntersectionLines += 1
                            if nIntersectionLines == 3:
                                print("Found intersection.")
                                return True
                            # third: int = self.findVerticalThird(y1, height)
                            # if third == 0:
                            #     nIntersectionLines += 1
                            #     if nIntersectionLines == 3:
                            #         print("Found intersection in top third.")
                            #         return True
                            # if third == 1:
                            #     nIntersectionLines += 1
                            #     if nIntersectionLines == 3:
                            #         print("Found intersection in middle third.")
                            #         return True
                            # if third == 2:
                            #     nIntersectionLines += 1
                            #     if nIntersectionLines == 3:
                            #         print("Found intersection in bottom third.")
                            #         return True
                # print("No intersection found.")
                return False
            else:
                # print("No intersection found.")
                return False
        print("No lines found.")
        return False

    def handleIntersection(self):
        """Handle what to do at an intersection.

        This function checks and updates global state to determine how to proceed
        when it's called. If the global state indicates that the robot must turn,
        this function handles that turn.

        This function intentionally blocks the main thread because the robot
        must stop and pivot, as opposed to turning while moving.
        """
        if self.path.turn:
            printing("Turning")
            self.stopMotors()
            self.turn()
            self.go()
        else:
            pass

    def setRightMotorSpeed(self, multiplier):
        currentSpeed = self.baseSpeed * 1.25 * multiplier

    def setLeftMotorSpeed(self, multiplier):
        currentSpeed = self.baseSpeed * 1.25 * multiplier

    def updateDirection(self, image, original):
        """Update the robot's direction given the orientation of the black line.

        Detect contours of black area, find centroid, and update motor speed.
        """
        # print("Updating direction.")
        contours, hierarchy = cv2.findContours(image.copy(), 1, cv2.CHAIN_APPROX_NONE)

        halfWidth = image.shape[1] / 2

        if len(contours) > 0:
            # Find largest contour area and its image moments
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)

            # Find x-coordinate of the centroid using image moments
            cx = int(M["m10"] / M["m00"])

        error = cx - halfWidth
        normalizedError = abs(error) / halfWidth
        if cx > halfWidth:  # Line on the right
            rightMotorMultiplier = 1 + normalizedError
            leftMotorMultiplier = 1 - normalizedError
        else:
            rightMotorMultiplier = 1 - normalizedError
            leftMotorMultiplier = 1 + normalizedError

        self.setRightMotorSpeed(rightMotorMultiplier)
        self.setLeftMotorSpeed(leftMotorMultiplier)

    def cvShowAndWait(self, image, title="Image"):
        """Show an image with cv2, wait for keypress "q", and close the window."""
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)  # Resizable window
        cv2.imshow(title, image)
        while True:
            if cv2.waitKey(self, 1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

    def run(self):
        try:
            while True:
                self.go()

                # Take photo with PiCam2
                rawImage = self.cam.capture_array()
                img = self.preprocess(rawImage)

                # Find intersections
                if self.findIntersection(img):
                    self.path.updateIntersection()
                    self.handleIntersection()

                # Update direction
                self.updateDirection(img, rawImage)

                # Allow processing time before repeating
                time.sleep(self.PROCESS_TIME)
        except KeyboardInterrupt:
            GPIO.cleanup()  # Clean up GPIO settings
