#!/usr/bin/env python3

import json
from Turn import Turn


class Path:
    """Stores global state representing the path that PiDust should take.
    The path can consist of following lines and turning at intersections
    of four lines."""

    def __init__(self, configFile="./path.json"):
        self.configFile = configFile

        # Whether to turn at the next intersection
        self.turn: bool = False
        # The next turn's direction
        self.turnCode = None
        # The number of intersections passed, i.e. the index of the next
        # intersection in the path.
        self.nIntersections: int = 0

        # Turns dictionary
        self.turns: Dict[int:Turn] = self.loadConfig()

        # Process config
        # self.processConfig()

    def updateIntersection(self):
        self.nIntersections += 1
        self.turn = self.nIntersections in self.turns
        if self.turn:
            self.turnCode = self.turns[self.nIntersections]

    def loadConfig(self):
        """Generate the turns dictionary from the config file."""
        with open(self.configFile, "r") as f:
            return json.load(f)

    # def processConfig(self):
    #     """Process config file."""
    #     for intersections, turn in
