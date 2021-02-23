"""
Global settings for game and program.

Functions
---------
getInfo() -> str
       Provides basic info about settings in str format.
getDictInfo() -> str
       Provides basic info about settings in dict format.
"""
import os

import psutil

title = "Snake AI"

# SCREEN AND MAP
screenSize = (600, 600)
mapSize = (12, 12)
area = mapSize[0] * mapSize[1]
order = 3

areaParam = area**order
perimeter = 2 * (mapSize[0] + mapSize[1])
coefficient, offset = 12, 50
a = coefficient * ((area-perimeter)/areaParam)
b = perimeter + offset

gridColors = ("lightBlue", "mediumBlue", "mediumBlue")  # color of GUI window background

# FPS AND DISPLAY
targetFPS = 120
smoothness = 3  # controls how fast and smooth animations run

# GENETICS
populationSize = 20
generations = 1000
displayTraining = False  # displays best snake after each generation during training

# SNAKE
initialSnakeSize = 4
maxSnakeVision = max(mapSize)  # how far rays are cast

def hungerFunc(size):
	#return 1000
    return min(a * (size**order) + b, mapSize[0] * mapSize[1]) + 1000

basicSnakeParams = {
    "initialSize": initialSnakeSize,
    "maxVision": maxSnakeVision,
}

#def calcHunger(size):
#    return min(a * (size**order) + b, mapSize[0] * mapSize[1])


	
smartShield = True  # allow behavior to overwrite AI neural network decisions
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
           "\n    Snake vision: " + str(maxSnakeVision) + \
           "\n    Smart shield: " + str(smartShield) + \
           "\n    Available CPU cores: " + str(cores)

def getDictInfo() -> dict:
    """
	Provides basic info in dict form.
    """
    return {
        "map size": mapSize,
        "population": populationSize,
        "generations": generations,
        "snake size": initialSnakeSize,
        "snake vision": maxSnakeVision,
        "smart shield": smartShield,
        "CPU cores": cores
    }
		   