#!/usr/bin/env python3

import cv2
import numpy as np
from Path import Path
import sys
import time

np.set_printoptions(threshold=np.inf)


class PiDust:
    def __init__(self):
        # Threshold
        self.THRESH = 150

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
        ret, thresh1 = cv2.threshold(blur, self.THRESH, 255, cv2.THRESH_BINARY_INV)
        resized = cv2.resize(thresh1, [24, 18])

        # Erode to eliminate noise, and dilate to restore eroded parts of image
        # mask = cv2.erode(thresh1, None, iterations=2)
        # mask = cv2.dilate(mask, None, iterations=2)
        return resized

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

    def updateDirection(self, image: list[list], original, livestream=False):
        """Update the robot's direction given the orientation of the black line.

        Compute column average of row average of position-weighted brightness of
        each row: C = sum(R) / height Use C to update direction.

        Recall that image is a list with length height, and with sublists of
        length width."""
        # print("Updating direction.")

        # self.cvShowAndWait(image)

        height, width = image.shape
        self.cvShowAndWait(image)

        # print(f"Image dimensions: {height}, {width}")
        positionArray = np.arange(0, width, 1)

        # Find avg position of white pixels
        rowWhiteAvgPosn = []
        for row in image:
            rowBrightnessNormalized = row / np.sum(row)
            rowWhiteAvgPosn.append(
                round(np.inner(positionArray, rowBrightnessNormalized))
            )
        print(f"Average position of white in row: { rowWhiteAvgPosn }")

        colWhiteAvgPosn = np.average(rowWhiteAvgPosn)
        delta: float = abs(colWhiteAvgPosn - width / 2)
        deltaPct = (delta / width / 2) * 100
        print(f"Average position of white: { colWhiteAvgPosn }")
        print(f"Delta: {delta}")
        print(f"Delta pct: {deltaPct}")
        print(f"Width: {width}")

        if delta < 5:
            print("Not updating direction")

    def cvShowAndWait(self, image, title="Image"):
        """Show an image with cv2, wait for keypress "q", and close the window."""
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)  # Resizable window
        cv2.imshow(title, image)
        while True:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

    def run(self):
        rawImage = cv2.imread(sys.argv[1])
        img = self.preprocess(rawImage)

        # Update direction
        self.updateDirection(img, rawImage)

        # self.cvShowAndWait(img)


if __name__ == "__main__":
    pidust = PiDust()
    pidust.run()
