"""
Exposes methods to play games, both hidden and rendered to GUI window.

Public Functions
---------
playGame
    Allows game to be played, calls appropriate sub function.
playTrainingGame
    Plays game with settings.mapSize sized map.
playPlayerGame
    Plays game with settings.mapSize sized map meant for player control.
"""

import numpy as np
import keyboard

from core import settings, environments, snakes, graphics, behaviors
from core.graphics import Engine

__author__ = "Grant Holmes"
__email__ = "g.holmes429@gmail.com"


def playGame(environment: environments.Environment, render: bool = True) -> None:
    """
    Allows game to be played, calls appropriate sub function.

    Parameters
    ----------
    environment: environments.Environment
        Environment for game
    render: bool, default=True
        Indicates if game should be rendered to GUI window
    """
    if render:
        _renderedGame(environment)
    else:
        _simulateGame(environment)


def playTrainingGame(snake: snakes.Snake, render: bool = False) -> dict:
    """
    Plays game with settings.mapSize sized map.

    Parameters
    ----------
    snake: snakes.Snake
        Snake to play game with
    render: bool, default=True
        Indicates if game should be rendered to GUI window
    """
    environment = environments.Environment(snake, settings.mapSize)
    playGame(environment, render=render)
    return {"fitness": snakes.Snake.fitness(snake),  "score": snake.score}
	
	
def playPlayerGame(environment: environments.Environment, spaceStart: bool = True) -> None:
    """
    Plays game with settings.mapSize sized map meant for player control.

    Parameters
    ----------
    snake: snakes.Snake
        Snake to play game with
    spaceStart: bool, default=True
        Indicates if game should be started with space or start automatically
    """
    _renderedGame(environment, spaceStart=spaceStart)


def _simulateGame(environment: environments.Environment) -> None:
    """
    Plays simulated game without GUI window

    Parameters
    ----------
    environment: environments.Environment
        Environment for game
    """
    while environment.active():
        environment.step()


def _renderedGame(environment: environments.Environment, spaceStart: bool = False) -> None:
    """
    Plays game rendered to GUI window.

    Parameters
    ----------
    environment: environments.Environment
        Environment for game
    """
    params = {
        "gridColors": settings.gridColors,
        "title": settings.title,
        "targetFPS": settings.targetFPS,
    }
    engine = Engine(settings.screenSize, environment.gameMap.size, checkered=True, **params)

    if spaceStart:
        while not any([keyboard.is_pressed(key) for key in ("space", "w", "a", "s","d", "up", "right", "down", "left")]) and engine.shouldRun():
            engine.clearScreen()
            engine.renderScene(_renderInitial, environment)
            engine.updateScreen()
	
    while environment.active():
        environment.step()
        if environment.active():
            for step in range(1, settings.smoothness + 1):
                if engine.shouldRun():
                    engine.clearScreen()
                    engine.renderScene(_renderEnvironment, environment, step)
                    engine.updateScreen()
    engine.exit()


# DEFINES HOW TO RENDER GAMES, WILL PASS TO ENGINE	

def _renderEnvironment(engine: graphics.Engine, environment: environments.Environment, step: int) -> None:
    """
    Defines how a Snake game environment is rendered to GUI window. Made to be passed into renderScene method of Engine.

    Parameters
    ----------
    engine: graphics.Engine
        Generic engine later converted to self when passed in to renderScene
    environment: environments.Environment
        Environment for game
    step: int
        Indicates progression into a single animation sequence
    """
    stepPercent = step / settings.smoothness
    prev, curr = np.array(environment.prevSnakeBody), np.array(environment.snake.body)  # Snake's body
    snakeColor = environment.snake.color
    headColor = tuple([min(255-value, 254) for value in snakeColor])  # min because can't render completely white circle for some reason

    origin = environment.snake.head
    motion = environment.snake.direction

    # render Snake's rays if Snake is controlled by AI
    if type(environment.snake.behavior) == behaviors.AI:
        for pair in environment.rays:

            unit = tuple([-1 if p < 0 else 1 if p > 0 else 0 for p in (pair[1][0] - origin[0], pair[1][1] - origin[1])])
            end = origin

            for step in range(settings.snakeVision):
                nextStep = (end[0] + unit[0], end[1] + unit[1])

                if -1 <= nextStep[0] <= environment.gameMap.size[0] and -1 <= nextStep[1] <= environment.gameMap.size[1]:
                    end = nextStep

            dx, dy = motion[0] * (stepPercent - 0.5) + 0.5, motion[1] * (stepPercent - 0.5) + 0.5
            start = engine.scaleUp((origin[0] + dx, origin[1] + dy))
            end = engine.scaleUp((end[0] + dx, end[1] + dy))

            engine.renderLine(start, end, 2, Engine.colors["red"])

    # render Snake's body
    for i, coord in enumerate(prev):
        engine.renderRect(engine.scaleUp(coord + stepPercent * (curr[i] - coord)), engine.paddedGridSize, snakeColor)
        if i > 1 and np.all(abs(coord - prev[i - 2]) == (1, 1)):
            engine.renderRect(engine.scaleUp(prev[i - 1]), engine.paddedGridSize, snakeColor)
        elif i == 1 and np.all(abs(curr[0] - prev[1]) == (1, 1)):
            engine.renderRect(engine.scaleUp(prev[0]), engine.paddedGridSize, snakeColor)

    # extra circle on head
    headScale = 0.67
    headCoefficient = 0.5 * headScale
    headOffset = 0.5 - headCoefficient
    engine.renderCircle(engine.scaleUp(prev[0] + stepPercent * (curr[0] - prev[0]) + headOffset), int(headCoefficient * engine.gridSize[0]), headColor)

    # if Snake just grew, render additional segment
    if len(prev) != len(curr):
        engine.renderRect(engine.scaleUp(curr[-1]), engine.paddedGridSize, snakeColor)

    # render food
    for coord in environment.gameMap.filter(1):
        engine.renderCircle(engine.scaleUp(coord), int(engine.gridSize[0] / 2), Engine.colors["red"])

    # render score
    txtPos = (environment.gameMap.size[0] * 0.8, environment.gameMap.size[1] * 0.075)
    engine.printToScreen("Score: " + str(environment.snake.score), engine.scaleUp(txtPos), 30, Engine.colors["blue"])
	
    # render hunger
    if environment.snake.starvation:
        txtPos = (environment.gameMap.size[0] * 0.8, environment.gameMap.size[1] * 0.125)
        engine.printToScreen("Hunger: " + str(min(int(abs(round(100 * environment.snake.hunger / environment.snake.maxHunger, 0))), 100)) + "%", engine.scaleUp(txtPos), 30, Engine.colors["blue"])
	

def _renderInitial(engine: graphics.Engine, environment: environments.Environment) -> None:
    """
    Defines how a Snake game environment is initially rendered to GUI window before game starts. Made to be passed into renderScene method of Engine.

    Parameters
    ----------
    engine: graphics.Engine
        Generic engine later converted to self when passed in to renderScene
    environment: environments.Environment
        Environment for game
    """
    curr = np.array(environment.snake.body)  # Snake's body
    snakeColor = environment.snake.color
    headColor = tuple([min(255-value, 254) for value in snakeColor])  # min because can't render completely white circle for some reason

    origin = environment.snake.head

    # render Snake's body
    for coord in curr:
        engine.renderRect(engine.scaleUp(coord), engine.paddedGridSize, snakeColor)

    # extra circle on head
    engine.renderCircle(engine.scaleUp(curr[0]), int(engine.gridSize[0] / 2), headColor)

    # render food
    for coord in environment.gameMap.filter(1):
        engine.renderCircle(engine.scaleUp(coord), int(engine.gridSize[0] / 2), Engine.colors["red"])

    # render score
    txtPos = (environment.gameMap.size[0] * 0.8, environment.gameMap.size[1] * 0.075)
    engine.printToScreen("Score: " + str(environment.snake.score), engine.scaleUp(txtPos), 30, Engine.colors["blue"])
	
    # render start prompt
    txtPos = (environment.gameMap.size[0] * 0.45, environment.gameMap.size[1] * 0.9)
    engine.printToScreen("Press space or movement keys...", engine.scaleUp(txtPos), 30, Engine.colors["blue"])
	
    # render hunger
    if environment.snake.starvation:
        txtPos = (environment.gameMap.size[0] * 0.8, environment.gameMap.size[1] * 0.125)
        engine.printToScreen("Hunger: " + str(min(int(abs(round(100 * environment.snake.hunger / environment.snake.maxHunger, 0))), 100)) + "%", engine.scaleUp(txtPos), 30, Engine.colors["blue"])
