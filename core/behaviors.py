"""
Contains behaviors for Snake that determine movement in response to Snake vision.

Classes
-------
Behavior
	Interface class with helper functions.
Manual
	Provides direction based on keyboard input.
AI
	Uses neural network to decide direction.
Replay
	Uses pre-recorded moves to decide direction.
"""

import keyboard
import numpy as np
from copy import deepcopy	

from core import neural_nets, searching
from core.constants import *

__author__ = "Grant Holmes"
__email__ = "g.holmes429@gmail.com"

# BEHAVIOR FACTORY
def getBehavior(behaviorType, *args, **kwargs):
	return {
		"hybrid": AI,
		"neural network": NeuralNetwork,
		"pathfinder": Pathfinder,
		"floodPathfinder": FloodPathfinder,
		"floodfill": FloodFill,
		"ghost": Replay,
		"player": Manual,
		"cycle": Hamiltonian,
	}[behaviorType](*args, **kwargs)

	
# BASE CLASSES

class Behavior:
	"""Interface class with helper functions."""
	def __init__(self) -> None:
		"""Does nothing, expandable"""
		if type(self) == Behavior:
			raise NotImplementedError

	def __call__(self) -> tuple:
		"""Does nothing"""
		raise NotImplementedError
		
	def getBrain(self):
		return {"type": "behavior"}
		
	def calcMoves(self, body, direction, awareness, environment):
		return
		
	def reset(self):
		return
		
	@staticmethod
	def getOrientedDirection(currDirection, newDirection, directionType) -> tuple:
		if not (abs(newDirection[0]) ^ abs(newDirection[1]) and abs(sum(newDirection)) == 1):
			raise ValueError("Invalid new direction")
		return {
			"local": lambda: {(-1, 0): Behavior.rotateCCW(currDirection), (0, 1): currDirection, (1, 0): Behavior.rotateCW(currDirection)},
			"global": lambda: {Behavior.rotateCCW(currDirection): (-1, 0), currDirection: (0, 1), Behavior.rotateCW(currDirection): (1, 0)}
		}[directionType]()[newDirection]

	@staticmethod
	def rotateCW(v):
		return (-v[1], v[0])
	
	@staticmethod
	def rotateCCW(v):
		return (v[1], -v[0])
	
# NON AI BEHAVIORS

class Manual(Behavior):
	"""Provides direction based on keyboard input."""
	def __init__(self) -> None:
		"""Initializes base class."""
		Behavior.__init__(self)
		self.body = None
		self.direction = None

	def __call__(self) -> tuple:
		"""
		Returns keyboard input, ignores vision.

		Parameters
		----------
		direction: tuple
			Current global direction Snake is facing

		Returns
		-------
		tuple: (new global direction, move necessary to have Snake oriented to this direction)
		"""
		if keyboard.is_pressed("w") or keyboard.is_pressed("up"):
			move = UP
		elif keyboard.is_pressed("a") or keyboard.is_pressed("left"):
			move = LEFT
		elif keyboard.is_pressed("s") or keyboard.is_pressed("down"):
			move = DOWN
		elif keyboard.is_pressed("d") or keyboard.is_pressed("right"):
			move = RIGHT
		else:
			move = self.direction

		newDirection = {False: move, True: self.direction}[move == (-self.direction[0], -self.direction[1])]
		move = self.getOrientedDirection(self.direction, newDirection, "global")

		return newDirection, move
		
	def calcMoves(self, body, direction, awareness, environment):
		self.body = body
		self.direction = direction

		
class Replay(Behavior):
	"""
	Uses pre-recorded moves to decide direction.

	Attributes
	----------
	t: int
		Indexes data
	"""
	def __init__(self, memories: list) -> None:
		"""
		Initializes base class.

		Parameters
		---------
		data: list
			List of (x, y) moves
		"""
		Behavior.__init__(self)
		self.memories = memories
		self.t = 0

	def __call__(self) -> tuple:
		"""
		Provides direction by indexing pre-recorded moves.

		Parameters
		----------
		vision: np.array
			Describes closeness of Snake's head to food, body, and wall
		direction: tuple
			Current global direction Snake is facing

		Returns
		-------
		tuple: (new global direction, move necessary to have Snake oriented to this direction)
		"""

		return self.nextDirection, self.nextMove
		
	def calcMoves(self, body, direction, awareness, environment):
		self.nextDirection = self.memories[self.t]
		self.nextMove = self.getOrientedDirection(direction, newDirection, "global")
		self.t += 1
	
class AI(Behavior):
	def __init__(
			self,
			ctrlLayers=(14, 16, 3),
			metaLayers=(6, 8, 5),
			shielded=True,
			ctrlWeights=None,
			ctrlBiases=None,
			metaWeights=None,
			metaBiases=None,
			):
		Behavior.__init__(self)
		self.dafaultOpenness = {(-1, 0): 0, (0, 1): 0, (1, 0): 0}
		self.openness = self.dafaultOpenness.copy()
		self.path = []
		self.ctrlNetwork = neural_nets.FFNN(ctrlLayers, weights=ctrlWeights, biases=ctrlBiases)
		self.metaNetwork = neural_nets.FFNN(metaLayers, weights=metaWeights, biases=metaBiases)
		self.shielded = True
		self.nextDirection, self.nextMove = None, None

	def getNetworkDecision(self, body, direction, vision):
		decision = np.argmax(self.ctrlNetwork.feedForward(vision))
		nextMove = {0: (-1, 0), 1: (0, 1), 2: (1, 0)}[decision]  # local direction
		if self.shielded:
			lethalMoves = {direction for direction, danger in zip([(-1, 0), (0, 1), (1, 0)], [vision[11] == 1 or vision[19] == 1, vision[8] == 1 or vision[16] == 1, vision[9] == 1 or vision[17] == 1]) if danger}
			prev = nextMove
			nextMove = self.smartShield(body[0], nextMove, lethalMoves)
		nextDirection = self.getOrientedDirection(direction, nextMove, "local")
		
		return nextDirection, nextMove
		
	def getOpenness(self, body, direction, environment):
		openness = {}
		for turnDirection in self.openness:
			newDirection = self.getOrientedDirection(direction, turnDirection, "local") 
			if environment[(probe := (body[0][0] + newDirection[0], body[0][1] + newDirection[1]))] != -1:  # if adjacent space is open
				openness[turnDirection] = searching.floodFillCount(deepcopy(environment), probe)
			else:
				openness[turnDirection] = 0
		return openness
		
	def getSafestMove(self, direction):
		if sum(self.openness.values()) != 0:
			nextMove = {0: (-1, 0), 1: (0, 1), 2: (1, 0)}[max(self.openness, key=self.openness.get)]
			nextDirection = self.getOrientedDirection(direction, nextMove, "local")
		else:
			nextDirection, nextMove = direction, (0, 1)  # straight
			
		return nextDirection, nextMove
		
	def getPath(self, environment, body, length: str):
		if not self.path:
			if length == "short":
				return searching.pathfind(environment, body[0], environment.filter(1)[0])[:-1]
			else:  # length == "long"
				return searching.longPathfind(environment, body[0], environment.filter(1)[0])[:-1]
			
	def getCycle(body, environment):
		if not self.path:
			if (initialPath := searching.longestPath(environment, body[0], body[-1], environment.filter(1)[0])[:-1]):
				connection = body[:-1]
				path = connection + initialPath
		return path
			
	def getMoveFromPath(self, body, direction):
		if self.path:
			moveTo = self.path.pop()
			nextDirection = (moveTo[0] - body[0][0], moveTo[1] - body[0][1])
			nextMove = self.getOrientedDirection(direction, newDirection, "global")
		else:
			newDirection, nextMove = direction, (0, 1)
		return newDirection, nextMove
		
	@staticmethod
	def smartShield(head, localDirection, lethalMoves):
		movesLeft = {(-1, 0), (0, 1), (1, 0)} - {localDirection}  # left, straight, right
		
		while movesLeft and localDirection in lethalMoves:
			localDirection = movesLeft.pop()
	
		return localDirection
		
# AI BEHAVIORS
class NeuralNetwork(AI):
	"""
	Uses neural network to decide direction.

	Attributes
	----------
	confidence: list
		Log of 'confidence' of decisions
	"""
	def __init__(self, **kwargs) -> None:
		"""
		Initializes base class, instantiates FFNN with layer sizes.

		Parameters
		----------
		layers: tuple, default=(24, 16, 3)
			Neural network layer architecture
		smartShield: bool, default=False
			Determines whether dangerous moves can be overwritten
			
		**kwargs
			Neural network super class parameters
		"""
		AI.__init__(self, **kwargs)

	def getBrain(self):
		return {"type": "neural network", "weights": self.network.weights, "biases": self.network.biases}

	def calcMoves(self, body, direction, awareness, environment):
		vision, visionBounds = searching.castRays(body[0], direction, environment, awareness["maxVision"])
		self.nextDirection, self.nextMove = self.getNetworkDecision(body, direction, vision)
		return {"visionBounds": visionBounds}
		
	def __call__(self) -> tuple:
		"""
		Provides direction by feeding vision to neural network.

		Parameters
		----------
		vision: np.array
			Describes closeness of Snake's head to food, body, and wall
		direction: tuple
			Current global direction Snake is facing

		Returns
		-------
		tuple: (new global direction, move necessary to have Snake oriented to this direction)
		"""

		#move = {0: "left", 1: "straight", 2: "right"}[decision] 
		 
		return self.nextDirection, self.nextMove
		
		
# MOVE STUFF LIKE OPENESS AND PATH INTO AI BASE
# ...WHICH ALLOWS STATIC METHODS TO BE CONVERTED TO SELF METHODS
# MAYBE MAKE AI ABLE TO WORK ALONE, USE STR ARGS TO SPECIFY BEHAVIOR TYPE? MIGHT MAKE HYBRID EASIER AT SAME TIME
# MIGRATE HAMILTONIAN AND FLOOD PATHFINDER
# MAKE HYBRID
# TEST
# FIX RECURSION DEPTH
# DOUBLE CHECK STRAIGHT; (0, 1) or (0, -1)??
# TEST TRAINING
# IF NOT WORKING DOUBLE CHECK ROTATION
# MIGHT STILL NEED TO MODIFY ENVIRONMENT -> SNAKE -> BEHAVIOR CONTORL FLOW
# EVENTUALLY REFACTOR BC VERY MESSY
# DOUBLE CHECK NOTES ON PHONE
# ... YOU GOT THIS!!!
		
class Pathfinder(AI):
	def __init__(self):
		AI.__init__(self)
		
	def calcMoves(self, body, direction, awareness, environment):
		self.path = self.getPath(environment, body, "short")
		self.nextDirection, self.nextMove = getMoveFromPath(self.path, body, direction)
		return {"path": set(self.path)}

	def __call__(self):
		return self.nextDirection, self.nextMove
		
class LongPathfinder(AI):
	def __init__(self):
		AI.__init__(self)
		
	def calcMoves(self, body, direction, awareness, environment):
		self.path = self.getPath(environment, body, "long")
		self.nextDirection, self.nextMove = getMoveFromPath(self.path, body, direction)
		return {"path": set(self.path)}

	def __call__(self):
		return self.nextDirection, self.nextMove
		
		
class Hamiltonian(AI):
	def __init__(self):
		AI.__init__(self)
		
	def calcMoves(self, body, direction, awareness, environment):
		self.path = self.getCycle(body, environment)
		if self.path:
			self.nextDirection, self.nextMove = getMoveFromPath(self.path, body, direction)
		else:
			self.nextDirection, self.nextMove = self.getSafestMove(direction)
		return {"path": set(self.path)}

	def __call__(self):
		return self.nextDirection, self.nextMove
		
		
class FloodPathfinder(AI):
	def __init__(self):
		AI.__init__(self)
		
	def calcMoves(self, body, direction, awareness, environment):
		self.path = self.getPath(environment, body, "short")
		if self.path:
			self.nextDirection, self.nextMove = getMoveFromPath(self.path, body, direction)
		else:
			self.nextDirection, self.nextMove = self.getSafestMove(direction)
		return {"path": set(self.path), "openness": self.openness}

	def __call__(self):
		return self.nextDirection, self.nextMove
		
class FloodFill(AI):
	def __init__(self):
		AI.__init__(self)
		
	def calcMoves(self, body, direction, awareness, environment):
		self.openness = self.getOpenness(body, direction, environment)
		self.nextDirection, self.nextMove = self.getSafestMove(direction)
		return {"openness": self.openness}

	def __call__(self):
		return self.nextDirection, self.nextMove