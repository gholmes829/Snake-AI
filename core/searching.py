from queue import PriorityQueue
from numba import jit
from copy import deepcopy
import numpy as np
 
from core.constants import *
 
def castRays(origin, orientation, space, rayLength) -> list:
	"""
	Cast octilinear rays out from Snake's head to provide Snake awareness of its surroundings.

	Note
	----
	'Closeness' defined as 1/dist.
	"""
	limits, rays = {}, {}

	bounds = {
		UP: origin[1],
		RIGHT: (space.size[0] - origin[0] - 1),
		DOWN: (space.size[1] - origin[1] - 1),
		LEFT: origin[0]
	}  # get distance from Snake's head to map borders

	
	for direction in ORTHOGONAL:  # determine how far rays can go
		limits[direction] = bounds[direction]

	for diagonal in DIAGONAL:
		limits[diagonal] = min(limits[(diagonal[0], 0)], limits[(0, diagonal[1])])

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

	for i, direction in enumerate(DIRECTIONS):  # for each direction
		for j, item in ((0, "food"), (8, "body"), (16, "wall")):
			vision[i + j] = rays[localizeDirection(orientation, direction)][item]  # add data, need to change reference so 'global up' will be 'Snake's left' is Snake if facing 'global right'

	# PRINT VALUES OF DATA TO DEBUG
	#for i in range(3):
	#    for j in range(8):
	#        print(round(data[i * 8 + j], 3), end=" ")
	#    print()
	return vision, visionBounds

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
	return {
		UP: lambda unit: unit,
		RIGHT: lambda unit: (-unit[1], unit[0]),
		DOWN: lambda unit: (-unit[0], -unit[1]),
		LEFT: lambda unit: (unit[1], -unit[0]),
	}[basis](direction)

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
	return ((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2) ** 0.5\
	
def pathfind(space, origin, target, impassable=-1) -> list:
	"""
	A* pathfinding with Manhattan distance for h cost
	"""
	path = []
	passable = {coord for coord, value in space.items() if value != impassable}
	if target not in passable or origin == target:  # target can not be reached or is already reached
		return path
	
	directions = {(0, -1), (1, 0), (0, 1), (-1, 0)}
	added = 0

	frontier = PriorityQueue()  # uses priority queue rather than iterative approach to increase performance
	cameFrom, costSoFar = {origin: origin}, {origin: 0}
	
	frontier.put((0, 0, origin))

	while not frontier.empty() and (current := frontier.get()[2]) != target:
		for direction in directions:
			if (neighbor := (current[0] + direction[0], current[1] + direction[1])) in passable:
				cost = costSoFar[current] + 1
				if neighbor not in costSoFar or cost < costSoFar[neighbor]:
					cameFrom[neighbor] = current
					costSoFar[neighbor] = cost
					priority = cost + abs(neighbor[0] - target[0]) + abs(neighbor[1] - target[1])  # Manhattan distance
					frontier.put((priority, added, neighbor))  # counter acts as tiebreaker in case costs are same
					added += 1

	if target not in cameFrom:  # target not reached
		return path

	path.append(target)
	while (add := cameFrom[path[-1]]) != cameFrom[add]:
		path.append(add)

	path.append(origin)
	return path
	
def floodFillCount(space, origin, stopValue=-1):
    #print(origin)
    if origin not in space or space[origin] == stopValue:
        return 0
    space[origin] = stopValue
    return 1 + sum([floodFillCount(space, (origin[0] + newMove[0], origin[1] + newMove[1])) for newMove in {(0, -1), (1, 0), (0, 1), (-1, 0)}])
	
	
	
	
def hamiltonian(space, origin, impassable=-1):
	numVertices = len([value for value in space.values() if value == 0]) + 2
	path = [origin] + [(-1, -1)] * (numVertices - 2)
	if cycle(space, path, 1):
		return path
	else:
		return []
	
def cycle(space, path, index):
	#print()
	#print(path, index)
	if index == len(path):
		#return valid(space, path[0], index, path[-1])
		start, end = path[0], path[-1]
		diff = (abs(start[0] - end[0]), abs(start[1] - end[1]))
		return sum(diff) == 1
	else:
		origin = path[index-1]
		for newMove in {(0, -1), (1, 0), (0, 1), (-1, 0)}:
			nextPos = (origin[0] + newMove[0], origin[1] + newMove[1])
			#print(nextPos)
			if valid(space, path, index, nextPos):
				path[index] = nextPos
				if cycle(space, path, index + 1):
					return True
				path[index] = (-1, -1)
		return False
		
		
def valid(space, path, index, nextPos):
	if nextPos in path or nextPos not in space or space[nextPos] == -1:
		return False
	prev = path[index - 1]
	diff = (abs(prev[0] - nextPos[0]), abs(prev[1] - nextPos[1]))
	return sum(diff) == 1

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	