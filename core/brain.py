from queue import PriorityQueue
from numba import jit
import numpy as np

from core.constants import *


rotations = {}

def rotationGenerator(theta):
	"""Returns lambda func to rotate tuple CCW by theta degrees"""
	thetaRad = np.radians(theta)
	cosTheta = np.cos(thetaRad)
	sinTheta = np.sin(thetaRad)
	return lambda v: (
		round(v[0] * cosTheta - v[1] * sinTheta, 5),
		round(v[0] * sinTheta + v[1] * cosTheta, 5)
	)

for step in range(8):
	theta = 45 * step
	rotations[-theta] = rotationGenerator(360 - theta)
	rotations[theta] = rotationGenerator(theta)
	
def rotate(v, theta):
	return rotations[theta](v)
	
#  localGlobal -- know prev direction and turn about to make, want to know global direction
#  globalLocal --  # know prev direction and new global direction, want to know what turn you took to get to new global direction

orientedDirections = {}
tempLocal = lambda currDirection: {(-1, 0): rotate(currDirection, -90), (0, 1): currDirection, (1, 0): rotate(currDirection, 90), (0, -1): rotate(currDirection, 180)}

for curr in ORTHOGONAL:
	for new in ORTHOGONAL:
		orientedDirections["local", curr, new] = tempLocal(curr)[new]
		
for curr in ORTHOGONAL:
	possibleNew = {}
	tempNew = (
		(rotate(curr, -90), (-1, 0)),
		(curr, (0, 1)),
		(rotate(curr, 90), (1, 0)),
		(rotate(curr, 180), (0, -1)),
		(rotate(curr, 45), (1, 1)),
		(rotate(curr, -45), (-1, 1)),
		(rotate(curr, 135), (1, -1)),
		(rotate(curr, -135), (-1, -1)),
	)
	for key, value in tempNew:
		possibleNew[round(key[0], 0), round(key[1], 0)] = value 

	for new in DIRECTIONS:
		orientedDirections["global", curr, new] = possibleNew[new]		
	
def getOrientedDirection(currDirection, newDirection, directionType) -> tuple:
	return orientedDirections[directionType, currDirection, newDirection]
	
localizedDirections = {}
	
for curr in ORTHOGONAL:
	for new in DIRECTIONS:
		localizedDirections[curr, new] = getOrientedDirection(curr, new, "global")

#[print(key, "|", value, "\n") for key, value in orientedDirections.items()] 	
#[print(key, "|", value, "\n") for key, value in localizedDirections.items()] 
		
def localizeDirection(basis: tuple, direction: tuple) -> tuple:
	"""
	Reorients direction to perspective of basis.

	Parameters
	----------
	basis: tuple
		Local direction
	direction: tuple
		Global direction

	Returns
	-------
	tuple: reoriented direction.
	"""
	return localizedDirections[basis, direction]

class FFNN:
	"""
	Feed forward neural network without backpropogation.

	Public Methods
	--------------
	feedForward(inputs) -> np.ndarray:
		Feeds inputs through neural net to obtain output.
	"""
	def __init__(self, layerSizes: list, activation: str = "sigmoid", weights: list = None, biases: list = None) -> None:
		"""
		Initializes.

		Parameters
		----------
		layerSizes: list
			Layer architecture
		activation: str, default="sigmoid"
			String denoting activation function to use
		weights: list, optional
			List of arrays of weights for each layer, randomized if not passed in
		biases: list, optional
			List of arrays of biases for each layer, randomized if not passed in
		"""
		activations = {
			"sigmoid": FFNN.sigmoid,
			"reLu": FFNN.reLu,
			"softmax": FFNN.softmax
		}
		self.layerSizes = layerSizes
		weightShapes = [(i, j) for i, j in zip(layerSizes[1:], layerSizes[:-1])]
		self.weights = [np.random.randn(*s) for s in weightShapes] if weights is None else weights
		self.biases = [np.random.standard_normal(s) for s in layerSizes[1:]] if biases is None else biases
		self.activation = activations[activation]

	def feedForward(self, inputs: np.ndarray) -> np.ndarray:
		"""
		Feeds inputs through neural net to obtain output.

		Parameters
		----------
		inputs: np.ndarray
			Inputs to neural network

		Returns
		-------
		np.ndarray: input after fed through neural network
		"""
		for w, b in zip(self.weights, self.biases):
			inputs = self.activation(inputs @ w.T + b)
		return inputs

	@staticmethod
	@jit(nopython=True)
	def sigmoid(x: float) -> float:
		"""Sigmoid."""
		return 1 / (1 + np.exp(-x))

	@staticmethod
	def reLu(x: float) -> float:
		"""Rectified linear unit."""
		return np.maximum(0, x)

	@staticmethod
	@jit(nopython=True)
	def softmax(v: np.ndarray) -> np.ndarray:
		"""Softmax probability output."""
		e = np.exp(v)
		return e / e.sum()

def castRays(origin, orientation, space, rayLength) -> list:
	"""
	Cast octilinear rays out from Snake's head to provide Snake awareness of its surroundings.

	Note
	----
	'Closeness' defined as 1/dist.
	"""
	rays = {}

	bounds = {
		UP: origin[1],
		RIGHT: (space.size[0] - origin[0] - 1),
		DOWN: (space.size[1] - origin[1] - 1),
		LEFT: origin[0]
	}  # get distance from Snake's head to map borders

	limits = {direction: bounds[direction] for direction in ORTHOGONAL}  # determine how far rays can go
	limits.update({diagonal: min(limits[(diagonal[0], 0)], limits[(0, diagonal[1])]) for diagonal in DIAGONAL})
	
	for direction in DIRECTIONS:  # determine closeness of Snake to walls, initialize rays dict
		distance = limits[direction] + 1 if direction in ORTHOGONAL else (limits[direction] + 1) * 1.414
		rays[direction] = {"wall": 1 / distance * int(distance <= rayLength), "food": 0, "body": 0}

	visionBounds = []
	probe = None
	for ray, targets in rays.items():  # ...in each 8 octilinear directions
		bound = min(limits[ray], rayLength)
		step = 1
		while not targets["food"] and not targets["body"] and step <= bound:  # take specified number of steps away from Snake's head and don't let rays search outside of map borders
			probe = (origin[0] + ray[0] * step, origin[1] + ray[1] * step)  # update probe position
			if not targets["food"] and space[probe] == FOOD:  # if food not found yet and found food
				targets["food"] = 1 / dist(origin, probe)
			elif not targets["body"] and space[probe] == DANGER:  # if body not found yet and found body
				targets["body"] = 1 / dist(origin, probe)
			step += 1	
		visionBounds.append((origin, (origin[0] + ray[0] * bound, origin[1] + ray[1] * bound)))  # add end of ray to list

	vision = np.zeros(24)

	for i, direction in enumerate(NORMAL_DIRECTIONS):  # for each direction
		for j, item in ((0, "food"), (8, "body"), (16, "wall")):
			vision[i + j] = rays[localizeDirection(orientation, direction)][item]  # add vision, need to change reference so 'global up' will be 'Snake's left' is Snake if facing 'global right'
	
	# TRY CHANGING DIST TO WITHOUT SQUAREROOT
	# VECTORIZE
	
	# PRINT VALUES OF VISION TO DEBUG
	#print("FOOD", "BODY", "WALL")
	#print(DIRECTIONS_STR)
	#for i in range(3):
	#    for j in range(8):
	#        print(round(vision[i * 8 + j], 3), end=" ")
	#    print()
	return vision, visionBounds

# NOTE THAT THE FOLLOWING IS DIVIDED BY 2
@jit(nopython=True)
def distOpt(pt1: tuple, pt2: tuple) -> float:
	"""
	Procides Euclidean distance, accelerated with jit.

	Parameters
	----------
	pt1: tuple
		First point
	pt2: tuple
		Second point

	Returns
	-------
	float: Euclidean distance
	"""
	return ((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2) / 2
	
@jit(nopython=True)
def dist(pt1: tuple, pt2: tuple) -> float:
	"""
	Procides Euclidean distance, accelerated with jit.

	Parameters
	----------
	pt1: tuple
		First point
	pt2: tuple
		Second point

	Returns
	-------
	float: Euclidean distance
	"""
	return ((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2) ** 0.5
	
def longPathfind(space, origin, target, impassable=-1) -> list:
	"""
	A* pathfinding.
	"""
	path = []
	passable = {coord for coord, value in space.items() if value != impassable}
	#if target not in passable or origin == target:  # target can not be reached or is already reached
	#	return path
	
	directions = {(0, -1), (1, 0), (0, 1), (-1, 0)}
	added = 0

	frontier = PriorityQueue()  # uses priority queue rather than iterative approach to increase performance
	cameFrom, costSoFar = {origin: origin}, {origin: 0}
	
	frontier.put((0, 0, origin))

	while not frontier.empty() and (current := frontier.get()[2]) != target:
		for direction in directions:
			if (neighbor := (current[0] + direction[0], current[1] + direction[1])) in passable or neighbor == target:
				cost = costSoFar[current] + 1
				if neighbor not in costSoFar or cost < costSoFar[neighbor]:
					cameFrom[neighbor] = current
					costSoFar[neighbor] = cost
					priority = 1 / (cost + dist(neighbor, target))  # inverted normal A* cost
					frontier.put((priority, added, neighbor))  # counter acts as tiebreaker in case costs are same
					added += 1

	if target not in cameFrom:  # target not reached
		return path

	path.append(target)
	while (add := cameFrom[path[-1]]) != cameFrom[add]:
		path.append(add)

	path.append(origin)
	return path

	

def pathfind(space, origin, target, impassable=-1) -> list:
	"""
	A* pathfinding.
	"""
	path = []
	passable = {coord for coord, value in space.items() if value != impassable}
	#if target not in passable or origin == target:  # target can not be reached or is already reached
	#	return path
	
	directions = {(0, -1), (1, 0), (0, 1), (-1, 0)}
	added = 0

	frontier = PriorityQueue()  # uses priority queue rather than iterative approach to increase performance
	cameFrom, costSoFar = {origin: origin}, {origin: 0}
	
	frontier.put((0, 0, origin))

	while not frontier.empty() and (current := frontier.get()[2]) != target:
		for direction in directions:
			if (neighbor := (current[0] + direction[0], current[1] + direction[1])) in passable or neighbor == target:
				cost = costSoFar[current] + 1
				if neighbor not in costSoFar or cost < costSoFar[neighbor]:
					cameFrom[neighbor] = current
					costSoFar[neighbor] = cost
					priority = cost + abs(neighbor[0] - target[0]) + abs(neighbor[1] - target[1])
					frontier.put((priority, added, neighbor))  # counter acts as tiebreaker in case costs are same
					added += 1

	if target not in cameFrom:  # target not reached
		return path

	path.append(target)
	while (add := cameFrom[path[-1]]) != cameFrom[add]:
		path.append(add)

	path.append(origin)
	return path
	
	

def longPathHelper(space, path, i, looseTarget, depth, impassable=-1, corner=True):
	allPaths = []
	hasIncremented = False
	while i < len(path) - 2:
		curr = path[i]
		next = path[i + 1]
		third = path[i + 2]
		extended = False
		fork = False
		direction1 = (next[0] - curr[0], next[1] - curr[1])
		direction2 = (third[0] - next[0], third[1] - next[1])
		if depth and (corner or hasIncremented) and all((direction1[0] + direction2[0], direction1[1] + direction2[1])):
			new = (curr[0] + direction2[0], curr[1] + direction2[1])
			if new in space and new not in path and space[new] != impassable:
				copy = path.copy()
				copy[i + 1] = new
				allPaths.append(longPathHelper(space, copy, i, looseTarget, depth-1, corner=False))

		turns = ((direction1[1], -direction1[0]), (-direction1[1], direction1[0]))
		for ti, turn in enumerate(turns):
			extension1, extension2 = (curr[0] + turn[0], curr[1] + turn[1]), (next[0] + turn[0], next[1] + turn[1])
			if extension1 in space and extension2 in space and extension1 not in path and extension2 not in path and space[extension1] != impassable and space[extension2] != impassable:
				path.insert(i + 1, extension1)
				path.insert(i + 2, extension2)
				extended = True
				fork = ti == 0
				break
		
		if fork and depth:
			extension1, extension2 = (curr[0] + turns[1][0], curr[1] + turns[1][1]), (next[0] + turns[1][0], next[1] + turns[1][1])
			if extension1 in space and extension2 in space and extension1 not in path and extension2 not in path and space[extension1] != impassable and space[extension2] != impassable:
				copy = path.copy()
				copy[i + 1] = extension1
				copy[i + 2] = extension2
				allPaths.append(longPathHelper(space, copy, i, looseTarget, depth - 1))
		hasIncremented = True	
			
		i += int(not extended)
	
	# last segment
	while i < len(path) - 1:
		curr = path[i]
		next = path[i+1]
		cornerPossible = False
		direction1 = (next[0] - curr[0], next[1] - curr[1])
		if i+2 < len(path):
			third = path[i + 2]
			direction2 = (third[0] - next[0], third[1] - next[1])
			cornerPossible = True
		extended = False
		fork = False
		
		if cornerPossible and depth and (corner or hasIncremented) and all((direction1[0] + direction2[0], direction1[1] + direction2[1])):
			new = (curr[0] + direction2[0], curr[1] + direction2[1])
			if new in space and new not in path and space[new] != impassable:
				copy = path.copy()
				copy[i + 1] = new
				allPaths.append(longPathHelper(space, copy, i, looseTarget, depth - 1, corner=False))
		
		turns = ((direction1[1], -direction1[0]), (-direction1[1], direction1[0]))
		for ti, turn in enumerate(turns):
			extension1, extension2 = (curr[0] + turn[0], curr[1] + turn[1]), (next[0] + turn[0], next[1] + turn[1])
			if extension1 in space and extension2 in space and extension1 not in path and extension2 not in path and space[extension1] != impassable and space[extension2] != impassable:
				path.insert(i + 1, extension1)
				path.insert(i + 2, extension2)
				extended = True
				fork = ti == 0
				break
		
		if fork and depth:
			extension1, extension2 = (curr[0] + turns[1][0], curr[1] + turns[1][1]), (next[0] + turns[1][0], next[1] + turns[1][1])
			if extension1 in space and extension2 in space and extension1 not in path and extension2 not in path and space[extension1] != impassable and space[extension2] != impassable:
				
				copy = path.copy()
				copy[i + 1] = extension1
				copy[i + 2] = extension2
				allPaths.append(longPathHelper(space, copy, i, looseTarget, depth - 1))
		hasIncremented = True

		i += int(not extended)

	allPaths.append(path)
	bestPath, bestLength = [], 0
	for path in allPaths:
		length = len(path)
		if length > bestLength or (looseTarget is not None and length == bestLength and looseTarget in path):
			bestPath = path
			bestLength = length
	
	return bestPath
	
	
def longestPath(space, origin, target, looseTarget=None, impassable=-1, depth=2, path = None) -> list:
	if path is None:
		path = pathfind(space, origin, target, impassable=impassable)
	if not path:
		return []
	longPath = longPathHelper(space, path.copy(), 0, looseTarget, depth, impassable=impassable)
	return longPath

	
# FLOODFILL COUNT
def floodFillCount(space, origin, impassable=-1, considerOrigin=True, depth=25):
	if not depth or (considerOrigin and (origin not in space or space[origin] == impassable)):
		return 0
	space[origin] = impassable
	return 1 + sum([floodFillCount(space, (origin[0] + newMove[0], origin[1] + newMove[1]), depth=depth-1) for newMove in {(0, -1), (1, 0), (0, 1), (-1, 0)}])
	
	
# HAMILTONIAN -- WAYYYYYYYYYYYY TOO SLOW
def hamiltonian(space, origin, target, impassable=-1):
	numVertices = len([value for value in space.values() if value == 0 or value == 1]) + 2
	path = [origin] + [(-1, -1)] * (numVertices - 2)
	success = cycleHelper(space, path, target, 1)
	if success:
		return path
	else:
		return []
	
def cycleHelper(space, path, target, index):
	if index == len(path):
		end = path[-1]
		diff = (abs(target[0] - end[0]), abs(target[1] - end[1]))
		return sum(diff) == 1
	else:
		origin = path[index-1]
		for newMove in {(0, -1), (1, 0), (0, 1), (-1, 0)}:
			nextPos = (origin[0] + newMove[0], origin[1] + newMove[1])
			if valid(space, path, index, nextPos):
				path[index] = nextPos
				if cycleHelper(space, path, target, index + 1):
					return True
				path[index] = (-1, -1)
		return False
		
		
def valid(space, path, index, nextPos):
	if nextPos in path or nextPos not in space or space[nextPos] == -1:
		return False
	prev = path[index - 1]
	diff = (abs(prev[0] - nextPos[0]), abs(prev[1] - nextPos[1]))
	return sum(diff) == 1

	

	
	
	
	
	
	
	
	
	
	
	
	
	