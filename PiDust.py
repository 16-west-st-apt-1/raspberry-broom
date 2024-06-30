#!/usr/bin/env python3

import cv2
import numpy as np
import sys
from Path import Path
from Turn import Turn


class PiDust:
    def __init__(self):
        # Lines with this angle wrto horizontal are treated as horizontal lines.
        self.MAX_HORIZ_LINE_ANGLE = 20

        # Minimum length of lines found with HoughLinesP transform.
        self.MIN_LINE_LENGTH = 100

        # Horizontal lines should be within these margin percentages of the image.
        self.LEFT_MARGIN: float = 0.15
        self.RIGHT_MARGIN: float = 1 - self.LEFT_MARGIN

        # Load Path
        self.path = Path()

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
        print("Checking for corners.")
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
            print("No intersection found.")
            return False
        else:
            print("No intersection found.")
            return False

    def leftTurn(self):
        """Turn left 90 degrees."""
        print("Turning left.")
        pass

    def rightTurn(self):
        """Turn right 90 degrees."""
        print("Turning right.")
        pass

    def turn(self):
        """Turn according to the turn type specified by Path."""
        turn = self.path.turnCode
        if turn == Turn.L:
            self.leftTurn()
        if turn == Turn.R:
            self.rightTurn()

    def handleIntersection(self):
        """Handle what to do at an intersection.

        This function checks and updates global state to determine how to proceed
        when it's called. If the global state indicates that the robot must turn,
        this function handles that turn.

        This function intentionally blocks the main thread because the robot
        must stop and pivot, as opposed to turning while moving.
        """
        if self.path.turn:
            # stop()
            self.turn()
            # go()
        else:
            pass

    def updateDirection(self, image, original):
        """Update the robot's direction given the orientation of the black line.

        Detect contours of black area, find centroid, and update motor speed.
        """
        print("Updating direction.")
        contours, hierarchy = cv2.findContours(image.copy(), 1, cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0:
            # Find largest contour area and image moments
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)

            # Find x-axis centroid using image moments
            cx = int(M["m10"] / M["m00"])

    def showAndWait(self, image, title="Image"):
        """Show an image with cv2, wait for keypress "q", and close the window."""
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)  # Resizable window
        cv2.imshow(title, image)
        while True:
            if cv2.waitKey(self, 1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

    def drawXLineAndShowAndWait(self, image, xcoord):
        """Draw a line at an x-coordinate on an image, and show it.
        The image is copied before drawing the line."""
        copy = image.copy()
        start_point = (xcoord, 0)
        end_point = (xcoord, image.shape[0])
        color = (0, 255, 0)  # green
        thickness = 3
        cv2.line(copy, start_point, end_point, color, thickness)
        showAndWait(copy)

    def run(self):
        # Parse args
        inputImage = sys.argv[1]

        rawImage = self.getImage(inputImage)

        img = self.preprocess(rawImage)

        if self.findIntersection(img):
            self.path.updateIntersection()
            self.handleIntersection()

        # updateDirection(img, rawImage)
        # sleep(0.1)
