"""
Snake objects that can play Snake game in an environment.

Classes
-------
Snake
	Snake base class.
SmartSnake
	Snake predetermined with neural net as behavior.
Player
	Snake predetermined with Manual player controller.
Ghost
	Snake with predetermined queue of moves.
"""

from random import choice

import numpy as np
from copy import deepcopy
from core.constants import *
from core import behaviors

__author__ = "Grant Holmes"
__email__ = "g.holmes429@gmail.com"


class Snake:
	"""
	Snake base class.

	Attributes
		----------
		initialSize: int
			Initial size of Snake's body
		vision: int
			Max number of steps Snake can see in
		size: int
			Number of times Snake gets food
		hunger: int
			How hungry Snake is, increases by 1 every time Snake moves, decreases by refeed when Snake eats
		age: int
			How old Snake is, increases by 1 every time Snake moves
		dead: bool
			If snake is dead or not
		body: list
			(x, y) coordinates of Snake's body
		head: tuple
			(x, y) coordinate of Snake's head
		direction: tuple
			(x, y) direction Snake is currently moving in
		prevTail: tuple
			(x, y) coordinate of previous pos of Snake's tail
		moveCount: dict
			Number of times Snake has moved in each direction
		kwargs: dict
			Saved copy of key word arguments so Snake can reset

	Public Methods
	--------------
	move(vision: list) -> None:
		Snake picks a direction to move and moves one step that direction.
	getBrain() -> Dict:
		Gets weights and biases of neural network if Snake controlled by AI.
	kill() -> None:
		Kills snake.
	revive() -> None:
		Resets Snake to allow Snake to play multiple games.
	grow() -> None:
		Increases body size by one, grows segment where tail used to be.
	setReference(origin: tuple) -> None:
		Moves Snake's coordinate frame of reference to new origin point.
	"""
	
	def __init__(self,
				 behaviorType: str,
				 behaviorArgs: list = None,
				 behaviorKwargs: dict = None,
				 initialSize: int = 4,
				 maxVision: int = 0,
				 hungerFunc: callable = lambda size: 0,
				 color: tuple = None,
				 ) -> None:
		"""
		Initializes.

		Parameters
		----------
		behavior: behaviors.behavior
			Called to determine Snake's next move based on Snake's vision
		starvation: bool, default=True
			Indicates whether Snake can starve to death or not
		initialSize: int, default=4
			Initial size of Snake's body
		vision: int, default=10
			Max number of steps Snake can see in
		maxHungerFunc: callable, default=lambda size: 200.
			Max hunger Snake can have before it dies
		refeedFunc: callable, default=lambda size: 150.
			How much Snake's hunger is reduces when it eats food
		color: tuple, optional
			Determines hunger of snake, color is random if not passed in
		"""
		behaviorArgs = [] if behaviorArgs is None else behaviorArgs  
		behaviorKwargs = {} if behaviorKwargs is None else behaviorKwargs
		
		if behaviorType == "neural network":
			self.fitness = self.controllerFitness
		else:
			self.fitness = self.metaFitness
		
		self.behavior = behaviors.getBehavior(behaviorType, *behaviorArgs, **behaviorKwargs)
 
		self.initialSize = initialSize
		self.size = initialSize
		self.hungerFunc = hungerFunc
		self.starvation = self.hungerFunc(initialSize)
		self.score = 0
		self.hunger = 0
		self.age = 0

		self.body = [(0, 0)] + [(-1 * (i + 1), 0) for i in range(self.size - 1)]
		self.head = self.body[0]
		self.prevTail = (-1 * self.size, 0)
		self.direction = (1, 0)
		
		self.color = choice(SNAKE_COLORS) if color is None else color

		self.moveTranslation = {(-1, 0): "left", (0, 1): "straight", (1, 0): "right"}
		self.moveCount = {"left": 0, "straight": 0, "right": 0}  # number of times Snake moves in each direction
		self.dead = False

		self.awareness = {
			"maxVision": maxVision,
			"visionBounds": [],
			"maxHunger": self.starvation,
			"path": [],
			"open": {(-1, 0): 0, (0, -1): 0, (1, 0): 0}
		}

	def move(self) -> None:
		"""
		Snake picks a direction to move and moves one step that direction.

		Parameters
		----------
		vision: list
			24x1 list of floats, 0-7 closeness to food, 8-15 closeness to body, 16-23 closeness to wall
		"""
		self.prevTail = self.body[-1]
		self.direction, move = self.behavior()
		
		self.moveCount[self.moveTranslation[move]] += 1

		self.body.pop()
		self.body.insert(0, (self.body[0][0] + self.direction[0], self.body[0][1] + self.direction[1]))
		self.head = self.body[0]
		
		if self.head in self.body[1:] or (self.starvation and self.hunger >= self.starvation):
			self.kill()
		else:
			self.hunger += 1
			self.age += 1

	def grow(self) -> None:
		"""Increases body size by one, grows segment where tail used to be."""
		self.size += 1
		self.score += 1
		self.body.append(self.prevTail)
		self.prevTail = (2 * self.prevTail[0] - self.body[-2][0], 2 * self.prevTail[1] - self.body[-2][1])
		self.starvation = self.hungerFunc(len(self))
		self.hunger = 0
		
	def navigate(self, environment):
		if (observed := self.behavior.calcMoves(self.body.copy(), deepcopy(self.direction), self.awareness, deepcopy(environment), self.hunger)) is not None:
			self.awareness.update(observed)
		
	def setReference(self, origin: tuple) -> None:
		"""
		Moves Snake's coordinate frame of reference to new origin point.

		Parameters
		----------
		origin: tuple
			Origin for new frame of reference
		"""
		self.body = [(origin[0] - self.head[0] + segment[0], origin[1] - self.head[1] + segment[1]) for segment in self.body]
		self.head = origin

	def kill(self) -> None:
		"""Kills snake."""
		self.dead = True
		
	def getBrain(self) -> dict:
		"""
		Gets weights and biases of neural network if Snake controlled by AI.

		Returns
		-------
		dict: dict("weights": neural net weights, "biases": neural net biases)

		Raises
		------
		TypeError: behavior doesn't have weights and biases if not AI based
		"""
		return self.behavior.getBrain()
		
	def reset(self):
		self.size = self.initialSize
		self.starvation = self.hungerFunc(self.size)
		self.body = [(0, 0)] + [(-1 * (i + 1), 0) for i in range(self.size - 1)]
		self.head = self.body[0]
		self.prevTail = (-1 * self.size, 0)
		self.direction = (1, 0)
		self.moveCount = {key:0 for key in self.moveCount.keys()}
		self.dead = False
		self.score = 0
		self.hunger = 0
		self.age = 0
		self.awareness = {
			"maxVision": self.awareness["maxVision"],
			"visionBounds": [],
			"maxHunger": self.starvation,
			"path": [],
			"open": {(-1, 0): 0, (0, -1): 0, (1, 0): 0}
		}
		self.behavior.reset()
		
	def __len__(self):
		"""Returns length of snake's body"""
		return self.size

	#@staticmethod
	#def fitness(snake) -> None:
		"""
		((snake_score)^3 * snake_age)/1000 + 1 if moved in all directions else 0
		"""
	#	return ((snake.score ** 3) * snake.age) / 1000 + 1 if all([p > 0 for p in snake.moveCount.values()]) else 0
		
	@staticmethod
	def metaFitness(snake) -> None:
		"""Testing..."""
		score = ((snake.score ** 3) * snake.age) / 1000 + 1 if sum([p > 0 for p in snake.behavior.algorithmCount.values()]) > 1 else 0
		#print(snake.behavior.algorithmCount, score)
		return score  # meta controller training
	
	@staticmethod
	def controllerFitness(snake) -> None:
		"""
		((snake_score)^3 * snake_age)/1000 + 1 if moved in all directions else 0
		"""
		return ((snake.score ** 3) * snake.age) / 1000 + 1 if all([p > 0 for p in snake.moveCount.values()]) else 0  # neural network snake training
	   
	@staticmethod
	def mergeTraits(child: object, parent1: object, parent2: object) -> None:
		"""
		Defines how Snake's traits are combined during genetic crossover.

		child: Snake
			Snake inheriting traits from parents
		parent1: Snake
			Parent whose traits are passed down to child
		parent2: Snake
			Parent whose traits are passed down to child
		"""
		child.color = ((parent1.color[0] + parent2.color[0]) / 2, (parent1.color[1] + parent2.color[1]) / 2, (parent1.color[2] + parent2.color[2]) / 2)
		
	# FACTORY METHODS
	@classmethod
	def Player(cls, **kwargs):
		return cls("player", hungerFunc=lambda size: 0, **kwargs)
		
	@classmethod
	def Ghost(cls, memories, **kwargs):
		return cls("replay", behaviorArgs=[memories], **kwargs)
		
