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

from core import settings
from core.game import environment, snakes, behaviors
from core.ui import graphics
from core.ui.graphics import Engine
from core.ui import render

__author__ = "Grant Holmes"
__email__ = "g.holmes429@gmail.com"


# consider adding try excecpt block to ensure engine always exits

def playGame(gameEnvironment: environment.Environment, render: bool = True) -> None:
	"""
	Allows game to be played, calls appropriate sub function.

	Parameters
	----------
	environment: environment.Environment
		Environment for game
	render: bool, default=True
		Indicates if game should be rendered to GUI window
	"""
	if render:
		_renderedGame(gameEnvironment)
	else:
		_simulateGame(gameEnvironment)


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
	#print("NEW GAME" + "!" * 20)  # delete
	# DONT MAKE NEW ENVIRONMENT EVERY TIME, MAKE ENVIRONMENT RESET
	gameEnvironment = environment.Environment(snake, settings.mapSize)
	playGame(gameEnvironment, render=render)
	id = snake.id
	return {"fitness": snake.fitness(snake), "score": snake.score, "id": id}
	
	
def playPlayerGame(gameEnvironment: environment.Environment, spaceStart: bool = True) -> None:
	"""
	Plays game with settings.mapSize sized map meant for player control.

	Parameters
	----------
	snake: snakes.Snake
		Snake to play game with
	spaceStart: bool, default=True
		Indicates if game should be started with space or start automatically
	"""
	_renderedGame(gameEnvironment, spaceStart=spaceStart)


def _simulateGame(gameEnvironment: environment.Environment) -> None:
	"""
	Plays simulated game without GUI window

	Parameters
	----------
	environment: environment.Environment
		Environment for game
	"""
	while gameEnvironment.active():
		gameEnvironment.step()


def _renderedGame(gameEnvironment: environment.Environment, spaceStart: bool = False) -> None:
	"""
	Plays game rendered to GUI window.

	Parameters
	----------
	gameEnvironment: environment.Environment
		Environment for game
	"""
	params = {
		"gridColors": settings.gridColors,
		"title": settings.title,
		"targetFPS": settings.targetFPS,
	}
	engine = Engine(settings.screenSize, gameEnvironment.gameMap.size, checkered=True, **params)

	if spaceStart:
		while not any([keyboard.is_pressed(key) for key in ("space", "w", "a", "s","d", "up", "right", "down", "left")]) and engine.shouldRun():
			engine.clearScreen()
			engine.renderScene(render._renderInitial, gameEnvironment)
			engine.updateScreen()
	
	while gameEnvironment.active():
		gameEnvironment.step()
		if gameEnvironment.active():
			for step in range(1, settings.smoothness + 1):
				if engine.shouldRun():
					engine.clearScreen()
					engine.renderScene(render._renderEnvironment, gameEnvironment, step)
					engine.updateScreen()
	engine.exit()
