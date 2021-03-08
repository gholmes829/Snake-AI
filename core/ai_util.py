"""

"""
from core import brain
import numpy as np

def getNetworkDecision(network, body, direction, vision):
	"""
	features = vision
	# ------------------------------------------------------------------------------------------------------------------------------------
	# ENSEMBLE-LIKE BAYESIAN ANALYSIS
	
	m = 50  # number of ensembles
	n = features.shape[0]  # number of input features
	
	# noise parameters based on normal distribution but could change model to multi modal or other distribution
	mu = 0  # normal mean
	sigma = 0.5  # normal std
	
	ensembles = np.zeros((m+1, n))
	
	ensembles[0] = features  # normal input
	for i in range(m):  # generate ensembles with noise
		ensembles[i+1] = features + np.random.normal(mu, sigma, n)  # mean, std, shape
	
	output = network.feedForward(ensembles)  # m x 3
	# would it be useful to take softmax of output to get probabilistic view?
	
	mu_hat = output.mean(axis=0)
	cov = np.cov(output, rowvar=False)  # get covariance matrix 3 x 3
	certainty = brain.FFNN.softmax((1 - cov.diagonal()))  # would this measure how much each decision varied with respect to pertubations in input?
	final = output[0] * certainty
	#equal = np.argmax(final) == np.argmax(out)
	#if not equal:
		#print(normal)
		#print(final)
		#print()
	#out = output[0]
	#normal = output[0]
	decision = np.argmax(final)
	#if np.argmax(normal) != decision:
		#print(out)
		#print(normal)
		#print(final)
		#print(np.argmax(normal), decision)
		#print()
	# ------------------------------------------------------------------------------------------------------------------------------------
	"""









	decision = np.argmax(network.feedForward(vision))
	
	
	
	
	
	
	
	
	
	
	
	
	nextMove = {0: (-1, 0), 1: (0, 1), 2: (1, 0)}[decision]
	nextDirection = brain.getOrientedDirection(direction, nextMove, "local")
	return nextDirection, nextMove
	
def getShieldedNetworkDecision(network, body, direction, vision):  # split into shielded and non shielded function options to avoid unnecesary conditional
	decision = np.argmax(network.feedForward(vision))
	nextMove = {0: (-1, 0), 1: (0, 1), 2: (1, 0)}[decision]  # local direction
	lethalMoves = {direction for direction, danger in zip([(-1, 0), (0, 1), (1, 0)], [vision[11] == 1 or vision[19] == 1, vision[8] == 1 or vision[16] == 1, vision[9] == 1 or vision[17] == 1]) if danger}
	nextMove = smartShield(body[0], nextMove, lethalMoves)
	nextDirection = brain.getOrientedDirection(direction, nextMove, "local")
	return nextDirection, nextMove
	
def smartShield(head, localDirection, lethalMoves):
	movesLeft = {(-1, 0), (0, 1), (1, 0)} - {localDirection}  # left, straight, right
	
	while movesLeft and localDirection in lethalMoves:
		localDirection = movesLeft.pop()

	return localDirection
	
def getPath(environment, body, length: str):
	if length == "short":
		return brain.pathfind(environment, body[0], environment.filter(1)[0])[:-1]
	else:  # length == "long"
		return brain.longPathfind(environment, body[0], environment.filter(1)[0])[:-1]


def getOpenness(body, direction, environment, depth=25):
	openness = {(-1, 0): 0, (0, 1): 0, (1, 0): 0}
	for turnDirection in openness:
		newDirection = brain.getOrientedDirection(direction, turnDirection, "local") 
		if environment[(probe := (body[0][0] + newDirection[0], body[0][1] + newDirection[1]))] != -1:  # if adjacent space is open
			openness[turnDirection] = brain.floodFillCount(environment.copy(), probe, depth=depth)  # might have to switch to deep copy
			
	return openness
		
def getSafestMove(openness, direction):
	nextMove = max(openness, key=openness.get)
	nextDirection = brain.getOrientedDirection(direction, nextMove, "local")
		
	return nextDirection, nextMove
	
def getClosestMove(distances, direction):
	nextMove = min(distances, key=distances.get)
	nextDirection = brain.getOrientedDirection(direction, nextMove, "local")
		
	return nextDirection, nextMove
	
def getCloseSafeMove(openness, distances, direction, threshold=1):
	safeMove = max(openness, key=openness.get)
	maxSafety = openness[safeMove]
	
	rewardMove = min(distances, key=distances.get)
	#print()
	#print(openness)
	#print(distances)
	if maxSafety > 0 and openness[rewardMove] / maxSafety >= threshold :
		#print("Reward")
		nextMove = rewardMove
	else:
		nextMove = safeMove
		#print("Safe")
		
	nextDirection = brain.getOrientedDirection(direction, nextMove, "local")
	return nextDirection, nextMove
	
def getMoveFromPath(path, body, direction):
	if path:
		moveTo = path.pop()
		nextDirection = (moveTo[0] - body[0][0], moveTo[1] - body[0][1])
		nextMove = brain.getOrientedDirection(direction, nextDirection, "global")
	else:
		nextDirection, nextMove = direction, (0, 1)
	return nextDirection, nextMove


def getDistances(origin, direction, target):
	distances = {(-1, 0): 0, (0, 1): 0, (1, 0): 0}
	for turnDirection in distances:
		newDirection = brain.getOrientedDirection(direction, turnDirection, "local") 
		probe = (origin[0] + newDirection[0], origin[1] + newDirection[1])
		distances[turnDirection] = brain.dist(probe, target)
	return distances
		
def getCycle(body, environment):
	if initialPath := brain.longestPath(environment, body[0], body[-1], environment.filter(1)[0])[:-1]:
		connection = body[:-1]
		full = connection + initialPath
		"""
		for i in range(len(connection) - 1):
			first = connection[i]
			second = connection[i+1]
			diff = (second[0]-first[0], second[1]-first[1])
			
			if diff not in {(1, 0), (-1, 0), (0, 1), (0, -1)}:
				print("Hole", i, first, second, diff)
				print("Body", body)
				print("Short path", brain.pathfind(environment, body[0], body[-1], impassable=impassable))
				raise AssertionError("Connection has holes", connection)
		"""
		#for i in range(len(initialPath) - 1):
		#	first = initialPath[i]
		#	second = initialPath[i+1]
		#	diff = (second[0]-first[0], second[1]-first[1])
		#	if diff not in {(1, 0), (-1, 0), (0, 1), (0, -1)}:
		#		print("Hole", i, first, second, diff)
		#		print("Body", body)
		#		print("Short path", brain.pathfind(environment, body[0], body[-1]))
		#		raise AssertionError("initialPath has holes", initialPath)
		"""
		for i in range(len(full) - 1):
			first = full[i]
			second = full[i+1]
			diff = (second[0]-first[0], second[1]-first[1])
			if diff not in {(1, 0), (-1, 0), (0, 1), (0, -1)}:
				print("Hole", i, first, second, diff)
				print("Body", body)
				print("Short path", brain.pathfind(environment, body[0], body[-1], impassable=impassable))
				raise AssertionError("full has holes", full)
		"""
		return full
	else:
		return []
		



	