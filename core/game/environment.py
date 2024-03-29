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
import numpy as np
from random import choice

from core.game import snakes
from core.game.constants import *

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

	def __init__(self, snake: snakes.Snake, mapSize: tuple, origin: tuple = None, food: list = None, noise: float = 0) -> None:
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
		self.origin = origin
		self.gameMap = Map(mapSize)
		self.noise = noise
		
		self.snake = snake
		self.prevSnakeBody = None
		if self.snake.dead:
			self.snake.reset()
			
		self.foodQueue = [] if food is None else food
		self.moveLog = []
		self.foodLog = []

		self.snakeWon = False
		
		self._placeSnake()
		self._placeFood()
		self.snake.navigate(self.gameMap, self.noise)

	def step(self) -> None:
		"""Takes a time step in game, calculating next state and updating game objects."""
		#print("\nBEFORE")
		#self.display()
       
		self.prevSnakeBody = self.snake.body.copy()
		self.snake.move()
		self.moveLog.append(self.snake.direction)
		entityAtHead = self.gameMap[self.snake.head]
		if entityAtHead == DANGER:  # Snake ran into wall or its body
			self.snake.kill()
		else:
			self.gameMap[self.snake.head] = DANGER
			if entityAtHead == FOOD:  # Snake ate food
				prevTail = self.snake.prevTail
				self.snake.grow()
				self.gameMap[prevTail] = DANGER
				self._placeFood()
			else:  # Snake moved into open space
				self.gameMap[self.snake.prevTail] = EMPTY
			#self.display()			
			self.snake.navigate(self.gameMap, self.noise)
		
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
			"moves": self.moveLog[::-1],
			"origin": self.origin,
			"food": self.foodLog[::-1],
			"mapSize": self.gameMap.size,
			"color": tuple(self.snake.color)
		}

	def display(self) -> None:
		"""Displays map and environment in terminal."""
		mapCopy = deepcopy(self.gameMap)
		for w in self.gameMap.filter(-1):
			mapCopy[w] = "+"
		for b in self.snake.body:
			mapCopy[b] = "#"
		mapCopy[self.snake.head] = "X"
		for pair in self.snake.awareness["visionBounds"]:
			mapCopy[pair[1]] = "@"
		mapCopy[self.gameMap.filter(1)[0]] = "f"
		for e in mapCopy.filter(0):
			mapCopy[e] = "."
		print(mapCopy)

	def _placeSnake(self) -> None:
		"""Translate Snake to its starting coordinates."""
		if self.origin is None:
			self.origin = self.snake.initialSize, int(self.gameMap.size[1] / 2)
		self.snake.setReference(self.origin)
		for coord in self.snake.body:
			self.gameMap[coord] = DANGER

	def _placeFood(self) -> None:
		"""Place food on map, either from predetermined pos or randomly chosen on open space."""
		pos = []
		if self.foodQueue:
			pos = self.foodQueue.pop()
		if not pos or self.gameMap[pos] != EMPTY:
			emptySpaces = self.gameMap.filter(EMPTY)
			if emptySpaces:
				pos = choice(emptySpaces)
				self.foodLog.append(pos)
				self.gameMap[pos] = FOOD
			else:
				self.snakeWon = True
				self.snake.kill()
				print("WIN!")
		else:
			self.foodLog.append(pos)
			self.gameMap[pos] = FOOD


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
	def __init__(self, size: tuple, inverted=True) -> None:
		"""
		Initializes map to target size.

		Parameters
		----------
		size: tuple
			Target map size
		"""
		dict.__init__(self)
		self.size = size
		self.inverted=inverted
		self.area = size[0] * size[1]
		edges = ({-1, size[0]}, {-1, size[1]})
		self.innerPerimeter = 2 * (size[0] + size[1])
		self.outerPerimeter = 2 * (size[0] + size[1]) + 4
		self.inverse = {}
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
		if self.inverted:
			return list(self.inverse[target])
		else:
			return [coord for coord, val in self.items() if val == target]
	
	
	def __setitem__(self, key, item):
		"""
		Handles inverse
		"""
		if self.inverted:
			if key in self:
				self.inverse[self[key]].remove(key)
			if item not in self.inverse:
				self.inverse[item] = set()
			self.inverse[item].add(key)
		super().__setitem__(key, item)		
	
	def __str__(self):
		return self.__repr__()

	def __repr__(self):
		output = ""
		for j in range(-1, self.size[1] + 1):
			for i in range(-1, self.size[0] + 1):
				next = "x" if self[(i, j)] == -1 else " " if self[(i, j)] == 0 else self[(i, j)]
				output += str(next)
			output += "\n"
		return output
