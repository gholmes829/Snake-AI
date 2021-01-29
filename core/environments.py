"""
Environment that modulates game logic with different settings.

Classes
-------
Environment
    Modulates game logic and flow.
Map
    Represents relative positions of objects in game.
"""

from copy import deepcopy
from numba import jit
import numpy as np
from random import choice

from core.constants import *
from core import snakes

__author__ = "Grant Holmes"
__email__ = "g.holmes429@gmail.com"


class Environment:
    """
    Modulates game logic and flow.

    Attributes
    ----------
    gameMap: Map
        Represents relative pos of game objects
    snake: snakes.Snake
        Snake that plays game
    origin: tuple
        Starting pos for Snake
    foodQueue: list
        If full, provides food positions to draw from
    moveLog: list
        Log of all moves made by Snake
    foodLog: list
        Log of all positions food placed on
    prevSnakeBody: list
        List of coords of Snake's body last time step
    rays: list
        List of (start, end) for rays generated when providing snakeVision
    snakeVision: list
        Flattened 8x3 list of Snake's closeness to food, its body, and walls

    Public Methods
    --------------
    step() -> None:
        Takes a time step in game, calculating next state and updating game objects.
    active() -> bool:
        Checks if game is over or not.
    getData() -> dict:
        Gets data from environment necessary to recreate game.
    display() -> None:
        Displays map and environment in terminal.
    """

    def __init__(self, snake: snakes.Snake, mapSize: tuple, origin: tuple = None, food: list = None) -> None:
        """
        Initializes environment, places snake and food.

        Parameters
        ----------
        snake: snakes.Snake
            Snake to play game in environment
        mapSize: tuple
            Size of map of game
        origin: tuple, optional
            Starting pos of Snake's head, placed center-left if not passed in
        food: list, optional
            List of food positions, if not passed in food placed randomly in open space
        """
        self.gameMap = Map(mapSize)
        self.snake = snake
        self.origin = origin

        if food is None:
            self.foodQueue = []
        else:
            self.foodQueue = food

        self.moveLog = []
        self.foodLog = []

        self.prevSnakeBody = []
        self.rays = []

        if self.snake.dead:
            self.snake.revive()

        self._placeSnake()
        self._placeFood()

        self.snakeVision = self._castRays()

    def step(self) -> None:
        """Takes a time step in game, calculating next state and updating game objects."""
        self.prevSnakeBody = self.snake.body.copy()
        self.snake.move(self.snakeVision)
        self.moveLog.append(self.snake.direction)

        valueAtHead = self.gameMap[self.snake.head]
        if valueAtHead == DANGER:  # Snake ran into wall or its body
            self.snake.kill()
        else:
            self.gameMap[self.snake.head] = DANGER
            if valueAtHead == FOOD:  # Snake ate food
                prevTail = self.snake.prevTail
                self.snake.grow()
                self.gameMap[prevTail] = DANGER
                self._placeFood()
            else:  # Snake moved into open space
                self.gameMap[self.snake.prevTail] = EMPTY
            self.snakeVision = self._castRays()  # update vision of Snake

    def active(self) -> bool:
        """
        Checks if game is over or not.

        Returns
        -------
        bool: if Snake is dead
        """
        return not self.snake.dead

    def getData(self) -> dict:
        """
        Gets data from environment necessary to recreate game.

        Returns
        -------
        dict: necessary information about environment
        """
        return {
            "moves": self.moveLog,
            "origin": self.origin,
            "food": self.foodLog[::-1],
            "mapSize": self.gameMap.size,
            "color": self.snake.color
        }

    def display(self) -> None:
        """Displays map and environment in terminal."""
        mapCopy = deepcopy(self.gameMap)
        for w in self.gameMap.filter(-1):
            mapCopy[w] = "+"
        for b in self.snake.body:
            mapCopy[b] = "#"
        mapCopy[self.snake.head] = "X"
        for pair in self.rays:
            mapCopy[pair[1]] = "@"
        mapCopy[self.gameMap.filter(1)[0]] = "f"
        for e in mapCopy.filter(0):
            mapCopy[e] = "."
        print(mapCopy)

    def _castRays(self) -> list:
        """
        Cast octilinear rays out from Snake's head to provide Snake awareness of its surroundings.

        Note
        ----
        'Closeness' defined as 1/dist.
        """
        origin = self.snake.head
        snakeDirection = self.snake.direction
        limits, rays = {}, {}

        # get distance from Snake's head to map borders
        bounds = {
            UP: origin[1],
            RIGHT: (self.gameMap.size[0] - origin[0] - 1),
            DOWN: (self.gameMap.size[1] - origin[1] - 1),
            LEFT: origin[0]
        }

        # determine how far rays can go
        for direction in ORTHOGONAL:
            limits[direction] = bounds[direction]

        for diagonal in DIAGONAL:
            limits[diagonal] = min(limits[(diagonal[0], 0)], limits[(0, diagonal[1])])

        # determine closeness of Snake to walls, initialize rays dict
        for direction in DIRECTIONS:
            distance = limits[direction] + 1 if direction in ORTHOGONAL else (limits[direction] + 1) * 1.414
            rays[direction] = {"wall": 1 / distance * int(distance <= self.snake.vision), "food": 0, "body": 0}

        self.rays.clear()  # reset so rays contains info only of this instance
        probe = None
        for step in range(1, self.snake.vision + 1):  # take specified number of steps away from Snake's head
            for ray, targets in rays.items():  # ...in each 8 octilinear directions
                if step <= limits[ray]:  # don't let rays search outside of map borders
                    probe = (origin[0] + ray[0] * step, origin[1] + ray[1] * step)  # update probe position
                    if targets["food"] == 0 and self.gameMap[probe] == FOOD:  # if food not found yet and found food
                        targets["food"] = 1 / Environment.dist(origin, probe)
                    elif targets["body"] == 0 and self.gameMap[probe] == DANGER:  # if body not found yet and found body
                        targets["body"] = 1 / Environment.dist(origin, probe)

                if step == min(self.snake.vision, limits[ray]):  # add end of ray to list
                    self.rays.append((origin, probe))

        data = [0 for _ in range(24)]

        for i, direction in enumerate(DIRECTIONS):  # for each direction
            for j, item in zip((0, 8, 16), ("food", "body", "wall")):
                # need to change reference so 'global up' will be 'Snake's left' is Snake if facing 'global right'
                data[j + i] = rays[Environment.changeReference(snakeDirection, direction)][item]  # add data

        # PRINT VALUES OF DATA TO DEBUG
        #for i in range(3):
        #    for j in range(8):
        #        print(round(data[i * 8 + j], 3), end=" ")
        #    print()
        #self.display()
        return np.array(data)

    def _placeSnake(self) -> None:
        """Translate Snake to its starting coordinates."""
        if self.origin is None:
            self.origin = self.snake.initialSize, int(self.gameMap.size[1] / 2)
        self.snake.translate(self.origin)
        for coord in self.snake.body:
            self.gameMap[coord] = DANGER

    def _placeFood(self) -> None:
        """Place food on map, either from predetermined pos or randomly chosen on open space."""
        pos = []
        if self.foodQueue:
            pos = (self.foodQueue.pop())
        if not pos or self.gameMap[pos] != EMPTY:
            pos = choice(self.gameMap.filter(EMPTY))

        self.foodLog.append(pos)
        self.gameMap[pos] = FOOD

    @staticmethod
    def changeReference(basis: tuple, direction: tuple) -> tuple:
        """
        Reorients direction to perspective of basis.

        Parameters
        ----------
        basis: tuple
            Local direction
        direction: tuple
            Global direction

        Returns
        -------
        tuple: reoriented direction.
        """
        return {
            UP: lambda unit: unit,
            RIGHT: lambda unit: (-unit[1], unit[0]),
            DOWN: lambda unit: (-unit[0], -unit[1]),
            LEFT: lambda unit: (unit[1], -unit[0]),
        }[basis](direction)

    @staticmethod
    @jit(nopython=True)
    def dist(pt1: tuple, pt2: tuple) -> float:
        """
        Procides Euclidean distance, accelerated with jit.

        Parameters
        ----------
        pt1: tuple
            First point
        pt2: tuple
            Second point

        Returns
        -------
        float: Euclidean distance
        """
        return ((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2) ** 0.5


class Map(dict):
    """
    Represents relative positions of objects in game.

    Attributes
    ----------
    size: tuple
        (x, y) map size

    Public Methods
    --------------
    filter(target: int) -> list:
        Provides coordinates of map that contain target value.
    """
    def __init__(self, size: tuple) -> None:
        """
        Initializes map to target size.

        Parameters
        ----------
        size: tuple
            Target map size
        """
        dict.__init__(self)
        self.size = size
        edges = ({-1, size[0]}, {-1, size[1]})
        for i in range(-1, size[0] + 1):
            for j in range(-1, size[1] + 1):
                if i in edges[0] or j in edges[1]:
                    self[(i, j)] = -1
                else:
                    self[(i, j)] = 0

    def filter(self, target: int) -> list:
        """
        Provides coordinates of map that contain target value.

        Parameters
        ----------
        target: int
            target value to return positions for

        Returns
        -------
        list: list of coordinates containing target value
        """
        return [coord for coord, val in self.items() if val == target]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        output = ""
        for j in range(-1, self.size[1] + 1):
            for i in range(-1, self.size[0] + 1):
                output += str(self[(i, j)])
            output += "\n"
        return output
