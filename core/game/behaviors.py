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

from core.game import brain, decisions
from core.game.constants import *

__author__ = "Grant Holmes"
__email__ = "g.holmes429@gmail.com"

# BEHAVIOR FACTORY
def getBehavior(behaviorType, *args, **kwargs):
	return {
		"neural network": NeuralNetwork,
		"multi": Multi,
		"hierarchical": Hierarchical,
		"pathfinder": Pathfinder,
		"floodfill": FloodFill,
		"cycle": Hamiltonian,
		"ghost": Replay,
		"player": Manual,
	}[behaviorType](*args, **kwargs)
		
		
# AI BEHAVIORS
class NeuralNetwork:
	def __init__(
			self,
			architecture = (24, 16, 3),
			weights = None,
			biases = None,
			shielded = False,
		) -> None:
		self.network = brain.FFNN(architecture, weights=weights, biases=biases)
		self.shielded = shielded
		self.nextDirection, self.nextMove = None, None
		self.getNetworkDecision = self.getNetworkDecisionFunc(self.shielded)
		self.ds = ((-1, 0), (0, -1), (1, 0))
		self.numOpenAdjacent = 0

	def calcMoves(self, body, direction, awareness, environment, hunger):
		vision, visionBounds = brain.castRays(body[0], direction, environment, awareness["maxVision"])
		self.nextDirection, self.nextMove = self.getNetworkDecision(self.network, body, direction, vision)
		head = body[0]
		adjacentDanger = sum([int(environment[head[0] + dx, head[1] + dy] == -1) for dx, dy in self.ds])
		self.numOpenAdjacent = 3 - adjacentDanger  # maybe change fitness heuristic to based on num danger instead of num open?
		return {"visionBounds": visionBounds}
		
	def __call__(self) -> tuple:
		return self.nextDirection, self.nextMove
	
	def getBrain(self):
		return {"type": "neural", "weights": self.network.weights, "biases": self.network.biases, "architecture": self.network.layerSizes}
	
	def reset(self):
		pass
	
	@staticmethod
	def getNetworkDecisionFunc(shielded):
		if shielded:
			return decisions.getShieldedNetworkDecision
		else:
			return decisions.getNetworkDecision

			
class Pathfinder:  # DEBUGGING FLOOD
	def __init__(self, pathLength = "short", floodfill=False):
		self.nextDirection, self.nextMove = None, None
		self.path = []
		self.openness = {(-1, 0): 0, (0, 1): 0, (1, 0): 0}
		self.pathLength = pathLength
		self.floodfill = floodfill
		
	def calcMoves(self, body, direction, awareness, environment, hunger):
		projected = set(self.path)
		if not self.path:
			self.path = decisions.getPath(environment, body, self.pathLength)
		if self.path:
			self.nextDirection, self.nextMove = decisions.getMoveFromPath(self.path, body, direction)
		elif self.floodfill:
			self.openness = decisions.getOpenness(body, direction, environment)
			self.nextDirection, self.nextMove = decisions.getSafestMove(self.openness, direction)
		else:
			self.nextDirection, self.nextMove = direction, (0, 1)
		return {"path": projected}

	def __call__(self):
		return self.nextDirection, self.nextMove

	def reset(self):
		self.path.clear()
		
	def brain(self):
		return {"type": "behavior"}
		
		
class Hamiltonian:
	def __init__(self, floodfill=False):
		self.nextDirection, self.nextMove = None, None
		self.path = []
		self.openness = None
		self.floodfill = floodfill
		
	def calcMoves(self, body, direction, awareness, environment, hunger):
		projected = set(self.path)
		if not self.path:
			self.path = decisions.getCycle(body, environment)
		if self.path:
			self.nextDirection, self.nextMove = decisions.getMoveFromPath(self.path, body, direction)
		elif self.floodfill:
			self.openness = decisions.getOpenness(body, direction, environment)
			self.nextDirection, self.nextMove = decisions.getSafestMove(self.openness, direction)
		else:
			self.nextDirection, self.nextMove = direction, (0, 1)
		return {"path": projected, "openness": self.openness}

	def __call__(self):
		return self.nextDirection, self.nextMove
		
	def reset(self):
		self.path.clear()
		
	def brain(self):
		return {"type": "behavior"}
		
class FloodFill:
	def __init__(self):
		self.nextDirection, self.nextMove = None, None
		self.openness = None
		self.distances = None
		
	def calcMoves(self, body, direction, awareness, environment, hunger):
		self.openness = decisions.getOpenness(body, direction, environment, depth=-1)
		self.distances = decisions.getDistances(body[0], direction, environment.filter(1)[0])
		self.nextDirection, self.nextMove = decisions.getCloseSafeMove(self.openness, self.distances, direction)
		return {"openness": self.openness}

	def __call__(self):
		return self.nextDirection, self.nextMove
		
	def reset(self):
		pass
			
	def brain(self):
		return {"type": "behavior"}
			
			
			
			
class Multi:  # genetic, pathfinding, floodfill
	def __init__(
			self,
			floodfill = True,
			architecture=(24, 16, 3),
			weights = None,
			biases = None,
			metaArchitecture=(21, 14, 3),
			metaWeights = None,
			metaBiases = None,
			shielded = False,
			pathLength = "short",
			):
		self.nextDirection, self.nextMove = None, None
		self.path = []
		self.openness = None
		self.shielded = shielded
		self.floodfill = floodfill
		self.pathLength = pathLength
		self.getNetworkDecision = NeuralNetwork.getNetworkDecisionFunc(self.shielded)
		
		self.network = brain.FFNN(architecture, weights=weights, biases=biases)
		self.metaNetwork = brain.FFNN(metaArchitecture, weights=metaWeights, biases=metaBiases)
		
		self.algoUsage = {"genetic": 0, "pathfind": 0, "floodfill": 0}
		self.algoIndexToName = {0: "genetic", 1: "pathfind", 2: "floodfill"}
		
		#self.prevSnakeSize = 0
		self.singleCounter = 0
		self.prevSnakeHunger = 0
		#self.maxSingleCounter = 5
		self.numOpenAdjacent = 0
		
		self.ds = ((-1, 0), (0, -1), (1, 0))

		# current decisions
		self.algoIndex = None
		self.algoName = None
		
	def calcMoves(self, body, direction, awareness, environment, hunger):
		# foodFound = False
		food = environment.filter(1)[0]
		head = body[0]
		tail = body[-1]
		
		#if len(body) != self.prevSnakeSize:
		#	foodFound = True
		#	self.prevSnakeSize = len(body)
			
		#if (self.algoIndex in {0, 2} and foodFound) or (self.algoIndex == 1 and not self.path) or self.algoIndex is None:
		if self.algoIndex in {0, 2} or (self.algoIndex == 1 and not self.path) or self.algoIndex is None:
			# calculate input features
			relativeSnakeSize = len(body) / environment.area
			
			foodCloseness = 1 / brain.distOpt(body[0], food)
			tailCloseness = 1 / brain.distOpt(body[0], body[-1])
			
			centerCloseness = 1 / (brain.distOpt(body[0], (int(environment.size[0]/2), int(environment.size[1]/2))) + 1)
				
			# danger as openness (need flood fill to search for openness, expensive)
			#leftSpace = self.openness[-1, 0] / environment.area
			#forwardSpace = self.openness[0, 1] / environment.area
			#rightSpace = self.openness[1, 0] / environment.area
			
			#proximity
			left, right = brain.rotate(direction, 90), brain.rotate(direction, -90)
			leftDanger = int(environment[head[0] + left[0], head[1] + left[1]] == -1)
			forwardDanger = int(environment[head[0] + direction[0], head[1] + direction[1]] == -1)
			rightDanger = int(environment[head[0] + right[0], head[1] + right[1]] == -1)
			
			self.numOpenAdjacent = 3 - leftDanger - forwardDanger - rightDanger
			
			leftDangerFood = int(environment[food[0] - 1, food[1]] == -1)
			upDangerFood = int(environment[food[0], food[1] - 1] == -1)
			rightDangerFood = int(environment[food[0] + 1, food[1]] == -1)
			downDangerFood = int(environment[food[0], food[1] + 1] == -1)
			
			leftDangerTail = int(environment[tail[0] - 1, tail[1]] == -1)
			upDangerTail = int(environment[tail[0], tail[1] - 1] == -1)
			rightDangerTail = int(environment[tail[0] + 1, tail[1]] == -1)
			downDangerTail = int(environment[tail[0], tail[1] + 1] == -1)
			
			# wall dist
			wallUp = 1 / (body[0][1] + 1)
			wallRight = 1 / (environment.size[0] - head[0] + 1)
			wallDown = 1 / (environment.size[1] - head[1] + 1)
			wallLeft = 1 / (head[0] + 1)
			
			# maybe change hunger to curr instead of previous unless change behavior on food aquisition
			
			# food relations
			relativeHunger = self.prevSnakeHunger/awareness["maxHunger"]
			manhattanMovesToFood = (abs(food[0] - head[0]) + abs(food[1] - head[1])) / (sum(environment.size))
			
			features = np.array([
				relativeSnakeSize,
				foodCloseness,
				tailCloseness,
				centerCloseness,
				wallUp,
				wallRight,
				wallDown,
				wallLeft,
				leftDanger,
				forwardDanger,
				rightDanger,
				leftDangerFood,
				upDangerFood,
				rightDangerFood,
				downDangerFood,
				leftDangerTail,
				upDangerTail,
				rightDangerTail,
				downDangerTail,
				relativeHunger,
				manhattanMovesToFood
			])
			
			self.algoIndex = np.argmax(self.metaNetwork.feedForward(features))
			self.algoName = self.algoIndexToName[self.algoIndex]
			# print(algoName)
			
			#self.singleCounter = self.maxSingleCounter
			
		if self.algoIndex == 0:
			vision, visionBounds = brain.castRays(head, direction, environment, awareness["maxVision"])
			self.nextDirection, self.nextMove = self.getNetworkDecision(self.network, body, direction, vision)
		elif self.algoIndex == 1:
			if not self.path:
				self.path = decisions.getPath(environment, body, self.pathLength)
			if self.path:
				self.nextDirection, self.nextMove = decisions.getMoveFromPath(self.path, body, direction)
			elif self.floodfill:
				self.openness = decisions.getOpenness(body, direction, environment)
				self.nextDirection, self.nextMove = decisions.getSafestMove(self.openness, direction)
			else:
				self.nextDirection, self.nextMove = direction, (0, 1)
		elif self.algoIndex == 2:
			self.openness = decisions.getOpenness(body, direction, environment, depth=-1)
			self.distances = decisions.getDistances(head, direction, food)
			self.nextDirection, self.nextMove = decisions.getCloseSafeMove(self.openness, self.distances, direction)
		else:  # space to define more
			raise NotImplementedError
		
		self.algoUsage[self.algoName] += 1
		self.prevSnakeHunger = hunger
		#self.singleCounter -= 1
		
	def __call__(self):
		return self.nextDirection, self.nextMove
		
	def reset(self):
		self.algoUsage = {"genetic": 0, "pathfind": 0, "floodfill": 0}
		self.path.clear()
		self.algoIndex = None
		self.algoName = None
		self.prevSnakeHunger = 0
		self.numOpenAdjacent = 0
		#self.singleCounter = 0
		
	def getBrain(self):
		return {
			"type": "multi",
			"weights": self.metaNetwork.weights,
			"biases": self.metaNetwork.biases,
			"metaArchitecture": self.metaNetwork.layerSizes,
			"networkWeights": self.network.weights,
			"networkBiases": self.network.biases,
			"networkArchitecture": self.network.layerSizes,
		}
		
		
class Hierarchical:
	def __init__(
			self,
			networkData = None,
			metaArchitecture = (24, 16, 3),
			metaWeights = None,
			metaBiases = None,
			shielded = False,
			):
		# reset architecture, removed additional features for testing
		self.nextDirection, self.nextMove = None, None
		self.metaNetwork = brain.FFNN(metaArchitecture, weights=metaWeights, biases=metaBiases)
		self.shielded = shielded
		
		self.getNetworkDecision = NeuralNetwork.getNetworkDecisionFunc(self.shielded)
		self.networks = []
		
		if networkData is None:
			networkData = []
			
		for data in networkData:
			architecture = data["architecture"]
			weights = data["weights"]
			biases = data["biases"]
			self.networks.append(brain.FFNN(architecture, weights=weights, biases=biases))
		
		# current decisions
		self.algoIndex = None
		self.algoName = None
		
		self.prevSnakeHunger = 0
		
		self.numOpenAdjacent = 0
		
		self.algoUsage = {"network 1": 0, "network 2": 0, "network 3": 0}
		self.algoIndexToName = {0: "network 1", 1: "network 2", 2: "network 3"}
		
	def calcMoves(self, body, direction, awareness, environment, hunger):
		food = environment.filter(1)[0]
		head = body[0]
		tail = body[-1]
		
		# calculate input features
		relativeSnakeSize = len(body) / environment.area
		
		# REMOVE UNNECESARY FEATURES
		#foodCloseness = 1 / brain.distOpt(body[0], food)
		#tailCloseness = 1 / brain.distOpt(body[0], body[-1])
		
		#centerCloseness = 1 / (brain.distOpt(body[0], (int(environment.size[0]/2), int(environment.size[1]/2))) + 1)
			
		# danger as openness (need flood fill to search for openness, expensive)
		#leftSpace = self.openness[-1, 0] / environment.area
		#forwardSpace = self.openness[0, 1] / environment.area
		#rightSpace = self.openness[1, 0] / environment.area
		
		#proximity
		left, right = brain.rotate(direction, 90), brain.rotate(direction, -90)
		leftDanger = int(environment[head[0] + left[0], head[1] + left[1]] == -1)
		forwardDanger = int(environment[head[0] + direction[0], head[1] + direction[1]] == -1)
		rightDanger = int(environment[head[0] + right[0], head[1] + right[1]] == -1)
		
		self.numOpenAdjacent = 3 - leftDanger - forwardDanger - rightDanger
		
		#leftDangerFood = int(environment[food[0] - 1, food[1]] == -1)
		#upDangerFood = int(environment[food[0], food[1] - 1] == -1)
		#rightDangerFood = int(environment[food[0] + 1, food[1]] == -1)
		#downDangerFood = int(environment[food[0], food[1] + 1] == -1)
		
		#leftDangerTail = int(environment[tail[0] - 1, tail[1]] == -1)
		#upDangerTail = int(environment[tail[0], tail[1] - 1] == -1)
		#rightDangerTail = int(environment[tail[0] + 1, tail[1]] == -1)
		#downDangerTail = int(environment[tail[0], tail[1] + 1] == -1)
		
		# wall dist
		#wallUp = 1 / (body[0][1] + 1)
		#wallRight = 1 / (environment.size[0] - head[0] + 1)
		#wallDown = 1 / (environment.size[1] - head[1] + 1)
		#wallLeft = 1 / (head[0] + 1)
		
		# food relations
		relativeHunger = self.prevSnakeHunger/awareness["maxHunger"]
		manhattanMovesToFood = (abs(food[0] - head[0]) + abs(food[1] - head[1])) / environment.innerPerimeter
		
		vision, visionBounds = brain.castRays(head, direction, environment, awareness["maxVision"])
		
		"""
		features = np.array([
			relativeSnakeSize,
			foodCloseness,
			tailCloseness,
			centerCloseness,
			wallUp,
			wallRight,
			wallDown,
			wallLeft,
			leftDanger,
			forwardDanger,
			rightDanger,
			leftDangerFood,
			upDangerFood,
			rightDangerFood,
			downDangerFood,
			leftDangerTail,
			upDangerTail,
			rightDangerTail,
			downDangerTail,
			relativeHunger,
			manhattanMovesToFood
		])
		"""
		
		# features = np.concatenate((np.array([relativeSnakeSize, relativeHunger, manhattanMovesToFood]), vision))
		features = vision  # delete, trying using only og features
		
		"""
		# ------------------------------------------------------------------------------------------------------------------------------------
		# ENSEMBLE-LIKE BAYESIAN ANALYSIS
		
		m = 25  # number of ensembles
		n = features.shape[0]  # number of input features
		
		# noise parameters based on normal distribution but could change model to multi modal or other distribution
		mu = 0  # normal mean
		sigma = 0.075  # normal std
		
		ensembles = np.zeros((m+1, n))
		
		ensembles[0] = features  # normal input
		for i in range(m):  # generate ensembles with noise
			ensembles[i+1] = features + np.random.normal(mu, sigma, n)  # mean, std, shape
		
		output = self.metaNetwork.feedForward(ensembles)  # m x 3
		# would it be useful to take softmax of output to get probabilistic view?
		
		mu_hat = output.mean(axis=0)
		cov = np.cov(output, rowvar=False)  # get covariance matrix 3 x 3
		certainty = brain.FFNN.softmax((1 - cov.diagonal()))  # would this measure how much each decision varied with respect to pertubations in input?
		final = mu_hat * certainty
		#equal = np.argmax(final) == np.argmax(out)
		#if not equal:
			#print(normal)
			#print(final)
			#print()
		normal = brain.FFNN.softmax(out)
		self.algoIndex = np.argmax(final)
		# ------------------------------------------------------------------------------------------------------------------------------------
		"""
		
		
		self.algoIndex = np.argmax(self.metaNetwork.feedForward(features))
		self.algoName = self.algoIndexToName[self.algoIndex]
		#print(algoName)
		self.algoUsage[self.algoName] += 1
		
		#vision, visionBounds = brain.castRays(head, direction, environment, awareness["maxVision"])
		self.nextDirection, self.nextMove = self.getNetworkDecision(self.networks[self.algoIndex], body, direction, vision)
		
		self.prevSnakeHunger = hunger
		
	def __call__(self):
		return self.nextDirection, self.nextMove
		
	def reset(self):
		self.algoUsage = {"network 1": 0, "network 2": 0, "network 3": 0}
		self.algoIndex = None
		self.algoName = None
		self.prevSnakeHunger = 0
		self.numOpenAdjacent = 0
		
	def getBrain(self):
		return {
			"type": "hierarchical",
			"weights": self.metaNetwork.weights,
			"biases": self.metaNetwork.biases,
			"metaArchitecture": self.metaNetwork.layerSizes,
			"networks": [{"weights": network.weights, "biases": network.biases, "architecture": network.layerSizes} for network in self.networks]
		}
			
		
# NON AI BEHAVIORS
class Manual:
	"""Provides direction based on keyboard input."""
	def __init__(self) -> None:
		"""Initializes base class."""
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
		move = brain.getOrientedDirection(self.direction, newDirection, "global")

		return newDirection, move
		
	def calcMoves(self, body, direction, awareness, environmen, hunger):
		self.body = body
		self.direction = direction

	def getBrain(self):
		return {"type": "behavior"}
		
	def reset(self):
		pass
		
class Replay:
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
		
	def calcMoves(self, body, direction, awareness, environment, hunger):
		self.nextDirection = self.memories[self.t]
		self.nextMove = brain.getOrientedDirection(direction, self.nextDirection, "global")
		self.t += 1
		
	def getBrain(self):
		return {"type": "behavior"}
		
	def reset(self):
		pass
		