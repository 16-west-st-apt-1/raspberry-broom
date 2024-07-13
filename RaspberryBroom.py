#!/usr/bin/env python3

import cv2
import numpy as np
from Path import Path
import RPi.GPIO as GPIO
import sys
import time
from Turn import Turn


class RaspberryBroom:
    def __init__(self, camera, motors, fps=3):
        # Lines with this angle wrto horizontal are treated as horizontal lines.
        self.MAX_HORIZ_LINE_ANGLE = 20

        # Minimum length of lines found with HoughLinesP transform.
        self.MIN_LINE_LENGTH = 100

        # Horizontal lines should be within these margin percentages of the image.
        self.LEFT_MARGIN: float = 0.15
        self.RIGHT_MARGIN: float = 1 - self.LEFT_MARGIN

        # Baseline duty cycle for testing
        self.TEST_BASE_DC = 30
        self.baseSpeed = self.TEST_BASE_DC
        self.TEST_MULTIPLIER = 0.5
        self.TURN_TIME = 0.5

        # Time between taking a new image
        self.PROCESS_TIME = 0.2

        # Load Path
        self.path = Path()

        # Store camera
        self.cam = camera
        self.setUpLiveStream(fps)

        # Store and start motors
        self.leftMotor = motors["leftA"]
        self.rightMotor = motors["rightA"]
        self.startMotors()

    def setUpLiveStream(self, fps):
        """Set up live stream.

        Open window for images and store the delay, in ms, between image
        updates in the livestream."""
        cv2.namedWindow("Image", cv2.WND_PROP_TOPMOST)
        self.msDelay = int(1000 / fps)

    def startMotors(self):
        self.leftMotor.start(0)
        self.rightMotor.start(0)

    def setLeftMotorSpeed(self, dc):
        """Set the left motor's duty cycle."""
        print(f"New left dc: {dc}")
        # if dc < 20:
        #     self.leftMotor.ChangeDutyCycle(20)
        # else:
        self.leftMotor.ChangeDutyCycle(dc)

    def setRightMotorSpeed(self, dc):
        """Set the right motor's duty cycle."""
        print(f"New right dc: {dc}")
        # if dc < 20:
        #     self.rightMotor.ChangeDutyCycle(20)
        # else:
        self.rightMotor.ChangeDutyCycle(dc)

    def multiplyLeftMotorSpeed(self, multiplier):
        """Multiply the left motor's speed by a multiplier."""
        self.setLeftMotorSpeed(self.baseSpeed * multiplier * self.TEST_MULTIPLIER)

    def multiplyRightMotorSpeed(self, multiplier):
        """Multiply the right motor's speed by a multiplier."""
        self.setRightMotorSpeed(self.baseSpeed * multiplier * self.TEST_MULTIPLIER)

    def stopMotors(self):
        self.setLeftMotorSpeed(0)
        self.setRightMotorSpeed(0)

    def go(self):
        self.setLeftMotorSpeed(self.baseSpeed)
        self.setRightMotorSpeed(self.baseSpeed)

    def leftTurn(self):
        """Turn left 90 degrees."""
        print("Turning left.")
        self.setLeftMotorSpeed(0)
        time.sleep(self.TURN_TIME)
        self.setLeftMotorSpeed(self.baseSpeed)

    def rightTurn(self):
        """Turn right 90 degrees."""
        print("Turning right.")
        self.setRightMotorSpeed(0)
        time.sleep(self.TURN_TIME)
        self.setRightMotorSpeed(self.baseSpeed)

    def turn(self):
        """Turn according to the turn type specified by Path."""
        turn = self.path.turnCode
        if turn == Turn.L:
            self.leftTurn()
        if turn == Turn.R:
            self.rightTurn()

    def crop(self, image):
        # Calculate the height of each third
        third_height = image.shape[0] // 4

        # Define the cropping region for the middle third
        y_start = third_height
        y_end = 3 * third_height

        # Crop the middle third
        return image[y_start:y_end, :]

    def preprocess(self, image):
        """Preprocess the image before OpenCV analysis.

        Convert to RGB, then grayscale, then blur, then threshold, then erode,
        then dilate."""

        # Convert RGBA to RGB
        # rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Gray, blur, and threshold >100 -> black
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, thresh1 = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)

        # Erode to eliminate noise, and dilate to restore eroded parts of image
        # mask = cv2.erode(thresh1, None, iterations=2)
        # mask = cv2.dilate(mask, None, iterations=2)
        return thresh1

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

        This requires some fine-tuning based on the distance between camera and
        the floor, the resolution, and the width and shapes of the lines being
        tracked."""

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

        linesToDraw = []
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
                            linesToDraw.append(line)
                            nIntersectionLines += 1
                            if nIntersectionLines == 4:
                                print("Found intersection.")
                                # for line in linesToDraw:
                                #     x1, y1, x2, y2 = line[0]
                                #     cv2.line(edges, (x1, y1), (x2, y2), (255, 0, 0), 3)
                                # cv2.namedWindow("Intersection", cv2.WINDOW_NORMAL)  # Resizable window
                                # cv2.imshow("Intersection", edges)
                                # cv2.waitKey(self.msDelay) & 0xFF == ord("q")
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
        # print("No lines found.")
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

    def updateDirection(self, image, original, livestream=False):
        """Update the robot's direction given the orientation of the black line.

        Detect contours of black area, find centroid, and update motor speed.
        If livestream is True, draw the centroid x-coord on the original image
        and show it."""
        # print("Updating direction.")
        contours, hierarchy = cv2.findContours(image, 1, cv2.CHAIN_APPROX_NONE)

        # Draw contours on the original image
        # cv2.drawContours(original, contours, -1, (0, 255, 0), 2)  # -1 draws all contours, (0, 255, 0) is color, 2 is thickness

        # Display the image with contours
        # cv2.imshow('Contours', original)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        halfWidth = image.shape[1] / 2

        if len(contours) > 0:
            # Find largest contour area and its image moments
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)

            # Find x-coordinate of the centroid using image moments
            cx = int(M["m10"]) / int(M["m00"])

            # Draw centroid x-coord line and update livestream
            # cv2.line(original, (cx, 0), (cx, 640), (0, 255, 0), 3)
            # cv2.imshow("Image", original)
            # cv2.waitKey(self.msDelay) & 0xFF == ord("q")

            error = cx - halfWidth

            change = 10 * (error / halfWidth)
            self.setLeftMotorSpeed(self.baseSpeed - change)
            self.setRightMotorSpeed(self.baseSpeed + change)

            # kp = 0.025

            # self.setLeftMotorSpeed(self.baseSpeed - (error * kp))
            # self.setRightMotorSpeed(self.baseSpeed + (error * kp))

            # error = cx - halfWidth
            # normalizedError = abs(error) / halfWidth
            # if cx > halfWidth:  # Line on the right
            #     rightMotorMultiplier = 1 + normalizedError
            #     leftMotorMultiplier = 1 - normalizedError
            # else:
            #     rightMotorMultiplier = 1 - normalizedError
            #     leftMotorMultiplier = 1 + normalizedError

            # self.multiplyRightMotorSpeed(rightMotorMultiplier)
            # self.multiplyLeftMotorSpeed(leftMotorMultiplier)

    def cvShowAndWait(self, image, title="Image"):
        """Show an image with cv2, wait for keypress "q", and close the window."""
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)  # Resizable window
        cv2.imshow(title, image)
        while True:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

    def run(self):
        try:
            # self.go()
            # Primary event loop
            while True:
                # Take photo with PiCam2
                rawImage = self.cam.capture_array()
                crop = self.crop(rawImage)
                img = self.preprocess(crop)

                self.cvShowAndWait(img)

                # Find intersections
                # if self.findIntersection(img):
                #     self.path.updateIntersection()
                #     self.handleIntersection()

                # Update direction
                # self.updateDirection(img, crop)

                # Allow processing time before repeating
                # time.sleep(self.PROCESS_TIME)
        except KeyboardInterrupt:
            self.cam.stop()
            GPIO.cleanup()  # Clean up GPIO settings
