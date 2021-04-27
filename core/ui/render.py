"""
Defines how games should be rendered.
"""

import numpy as np

from core.ui import graphics
from core.ui.graphics import Engine
from core.game import environment
from core import settings

def _renderEnvironment(engine: graphics.Engine, gameEnvironment: environment.Environment, step: int) -> None:
	"""
	Defines how a Snake game environment is rendered to GUI window. Made to be passed into renderScene method of Engine.

	Parameters
	----------
	engine: graphics.Engine
		Generic engine later converted to self when passed in to renderScene
	environment: environment.Environment
		Environment for game
	step: int
		Indicates progression into a single animation sequence
	"""
	stepPercent = step / settings.smoothness
	prev, curr = np.array(gameEnvironment.prevSnakeBody), np.array(gameEnvironment.snake.body)  # Snake's body
	snakeColor = gameEnvironment.snake.color
	headColor = tuple([min(255-value, 254) for value in snakeColor])  # min because can't render completely white circle for some reason

	origin = gameEnvironment.snake.head
	motion = gameEnvironment.snake.direction

	# render Snake's rays if Snake casts rays
	for pair in gameEnvironment.snake.awareness["visionBounds"]:

		unit = tuple([-1 if p < 0 else 1 if p > 0 else 0 for p in (pair[1][0] - origin[0], pair[1][1] - origin[1])])
		end = origin

		for step in range(settings.maxSnakeVision):
			nextStep = (end[0] + unit[0], end[1] + unit[1])

			if -1 <= nextStep[0] <= gameEnvironment.gameMap.size[0] and -1 <= nextStep[1] <= gameEnvironment.gameMap.size[1]:
				end = nextStep

		dx, dy = motion[0] * (stepPercent - 0.5) + 0.5, motion[1] * (stepPercent - 0.5) + 0.5
		start = engine.scaleUp((origin[0] + dx, origin[1] + dy))
		end = engine.scaleUp((end[0] + dx, end[1] + dy))

		engine.renderLine(start, end, 2, Engine.colors["red"])
		
	for coord in gameEnvironment.snake.awareness["path"]:
		engine.renderRect(engine.scaleUp(coord), engine.gridSize, Engine.colors["red"], alpha=50)

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
	for coord in gameEnvironment.gameMap.filter(1):
		engine.renderCircle(engine.scaleUp(coord), int(engine.gridSize[0] / 2), Engine.colors["red"])

	# render score
	txtPos = (gameEnvironment.gameMap.size[0] * 0.8, gameEnvironment.gameMap.size[1] * 0.075)
	engine.printToScreen("Score: " + str(gameEnvironment.snake.score), engine.scaleUp(txtPos), 30, Engine.colors["blue"])
	
	# render hunger
	if gameEnvironment.snake.starvation:
		txtPos = (gameEnvironment.gameMap.size[0] * 0.8, gameEnvironment.gameMap.size[1] * 0.125)
		engine.printToScreen("Hunger: " + str(min(int(abs(round(100 * gameEnvironment.snake.hunger / gameEnvironment.snake.starvation, 0))), 100)) + "%", engine.scaleUp(txtPos), 30, Engine.colors["blue"])
	

def _renderInitial(engine: graphics.Engine, gameEnvironment: environment.Environment) -> None:
	"""
	Defines how a Snake game environment is initially rendered to GUI window before game starts. Made to be passed into renderScene method of Engine.

	Parameters
	----------
	engine: graphics.Engine
		Generic engine later converted to self when passed in to renderScene
	environment: environment.Environment
		Environment for game
	"""
	curr = np.array(gameEnvironment.snake.body)  # Snake's body
	snakeColor = gameEnvironment.snake.color
	headColor = tuple([min(255-value, 254) for value in snakeColor])  # min because can't render completely white circle for some reason

	origin = gameEnvironment.snake.head

	# render Snake's body
	for coord in curr:
		engine.renderRect(engine.scaleUp(coord), engine.paddedGridSize, snakeColor)

	# extra circle on head
	headScale = 0.67
	headCoefficient = 0.5 * headScale
	headOffset = 0.5 - headCoefficient
	engine.renderCircle(engine.scaleUp(curr[0] + headOffset), int(headCoefficient * engine.gridSize[0]), headColor)

	# render food
	for coord in gameEnvironment.gameMap.filter(1):
		engine.renderCircle(engine.scaleUp(coord), int(engine.gridSize[0] / 2), Engine.colors["red"])

	# render score
	txtPos = (gameEnvironment.gameMap.size[0] * 0.8, gameEnvironment.gameMap.size[1] * 0.075)
	engine.printToScreen("Score: " + str(gameEnvironment.snake.score), engine.scaleUp(txtPos), 30, Engine.colors["blue"])
	
	# render start prompt
	txtPos = (gameEnvironment.gameMap.size[0] * 0.45, gameEnvironment.gameMap.size[1] * 0.9)
	engine.printToScreen("Press space or movement keys...", engine.scaleUp(txtPos), 30, Engine.colors["blue"])
	
	# render hunger
	if gameEnvironment.snake.starvation:
		txtPos = (gameEnvironment.gameMap.size[0] * 0.8, gameEnvironment.gameMap.size[1] * 0.125)
		engine.printToScreen("Hunger: " + str(min(int(abs(round(100 * gameEnvironment.snake.hunger / gameEnvironment.snake.starvation, 0))), 100)) + "%", engine.scaleUp(txtPos), 30, Engine.colors["blue"])