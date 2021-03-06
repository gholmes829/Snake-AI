"""Constant variables to improve readability."""

__author__ = "Grant Holmes"
__email__ = "g.holmes429@gmail.com"

FOOD = 1
EMPTY = 0
DANGER = -1

UP = (0, -1)
RIGHT = (1, 0)
DOWN = (0, 1)
LEFT = (-1, 0)

UP_RIGHT = (1, -1)
UP_LEFT = (-1, -1)
DOWN_RIGHT = (1, 1)
DOWN_LEFT = (-1, 1)

# INCONSISTENET UP??
ORTHOGONAL = [UP, RIGHT, DOWN, LEFT]
NORMAL_ORTHOGONAL = [DOWN, RIGHT, UP, LEFT]
DIAGONAL = [UP_RIGHT, DOWN_RIGHT, DOWN_LEFT, UP_LEFT]
DIRECTIONS = [UP, RIGHT, DOWN, LEFT, UP_RIGHT, DOWN_RIGHT, DOWN_LEFT, UP_LEFT]
NORMAL_DIRECTIONS = [DOWN, RIGHT, UP, LEFT, DOWN_RIGHT, UP_RIGHT, UP_LEFT, DOWN_LEFT]
DIRECTIONS_STR = ["UP", "RIGHT", "DOWN", "LEFT", "UP_RIGHT", "DOWN_RIGHT", "DOWN_LEFT", "UP_LEFT"]

SNAKE_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (255, 255, 0),
    (255, 165, 0),
    (234, 10, 142),
    (0, 0, 0)
]
