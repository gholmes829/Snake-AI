"""
Global settings for game and program.

Functions
---------
getInfo() -> str
       Provides basic info about settings in str format.
"""
import os

import psutil

title = "Snake AI"

screenSize = (600, 600)
mapSize = (15, 15)

gridColors = ("lightBlue", "mediumBlue", "mediumBlue")  # color of GUI window background

targetFPS = 60
smoothness = 5  # controls how fast and smooth animations run

initialSnakeSize = 4
snakeVision = max(mapSize)  # how far rays are cast
maxHunger = 200
refeed = 175
snakeParams = {"initialSize": initialSnakeSize, "vision": snakeVision, "maxHunger": maxHunger, "refeed": refeed}
networkArchitecture = (24, 16, 3)  # FFNN layers

populationSize = 100
generations = 5
displayTraining = True  # displays best snake after each generation during training

cores = psutil.cpu_count(logical=False)  # number of CPU cores, not including logical processors
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"  # hide Pygame greeting message


def getInfo() -> str:
    """
    Provides basic info about settings in str format.
    """
    return "    Map size: " + str(mapSize) + \
           "\n    Population: " + str(populationSize) + \
           "\n    Target generations: " + str(generations) + \
           "\n    Starting snake size: " + str(initialSnakeSize) + \
           "\n    Snake vision: " + str(snakeVision) + \
           "\n    Max hunger: " + str(maxHunger) + \
           "\n    Refeed bonus: " + str(refeed) + \
           "\n    Available CPU cores: " + str(cores)
