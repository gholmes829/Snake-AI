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

# SCREEN AND MAP
screenSize = (600, 600)
mapSize = (15, 15)
area = mapSize[0] * mapSize[1]
order = 3
areaParam = area**order
perimeter = 2 * (mapSize[0] + mapSize[1])

gridColors = ("lightBlue", "mediumBlue", "mediumBlue")  # color of GUI window background

# FPS AND DISPLAY
targetFPS = 60
smoothness = 3  # controls how fast and smooth animations run

# GENETICS
populationSize = 500
generations = 250
displayTraining = False  # displays best snake after each generation during training

# SNAKE
initialSnakeSize = 4
snakeVision = max(mapSize)  # how far rays are cast

def calcHunger(size):
    return 10 * ((area-perimeter)/areaParam) * (size**order) + perimeter + 10

def calcMaxHunger(size):
    """Calculates how hungry quickly snake can starve based on its size"""
    return calcHunger(size)
    
def calcRefeed(size):
    """Calculates how much snake's hunger diminishes based on its size"""
    return calcHunger(size)

smartShield = True  # allow behavior to overwrite AI neural network decisions
snakeParams = {"initialSize": initialSnakeSize, "vision": snakeVision, "maxHunger": calcMaxHunger, "refeed": calcRefeed}
networkArchitecture = (24, 16, 3)  # FFNN layers

# HELPER
cores = psutil.cpu_count(logical=False)  # number of CPU cores, not including logical processors
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"  # hide Pygame greeting message

def getInfo() -> str:
    """
    Provides basic info about settings in str format.
    """
    return "    Map size: " + str(mapSize) + \
           "\n    Population: " + str(populationSize) + \
           "\n    Target generations: " + str(generations) + \
           "\n    Starting snake size: " + str(initialSnakeSize) + \
           "\n    Snake vision: " + str(snakeVision) + \
           "\n    Smart shield: " + str(smartShield) + \
           "\n    Available CPU cores: " + str(cores)
