@staticmethod
def h_cost(cell, target, method):
	"""h_cost used for A* and HPA* to compute dist bt cell and target, select chooses method"""
	dx = abs(cell.location.x - target.x)
	dy = abs(cell.location.y - target.y)

	if method == 0:  # this method will result in a straighter line
		return dx + dy
	else:  # this method results in a more zig-zag line
		if dx > dy:
			return dx
		else:
			return dy

def validMove(self, current, nextMove):
	"""checks if going from current to nextMove is valid by checking if walls block movement"""
	if nextMove in self.walls:
		return False

	diff = nextMove - current

	if diff.x == 0 or diff.y == 0:
		return True
	elif (current + Coord(0, diff.y) in self.walls) and current + Coord(diff.x, 0) in self.walls:
		return False
	else:
		return True

@staticmethod
def overlaps(pt, paths):
	"""tests to see if point is in paths, diversifies multi-agent pathfinding"""
	if paths is None:
		return False
	else:
		return paths[pt]

def search(self, start, target, searchType, paths=None, abort=None, altTargets=None):
	"""
	allows pathfinding through map around walls
	Parameters:
		start -> where to start search
		target -> target to pathfind to
		searchType -> A* or HPA*
		paths -> can pass in the projected paths of all enemies to reduce agent density and improve path diversity
		abort -> returns with failed path if searches this many cells, used to avoid long expensive paths
		altTargets -> if path fails to find target, checks if any in altTargets were found and returns path to them
	Search Types:
		A* -> heuristic based pathfinding with dynamic terrain costs
		HPA* -> extension of A* with levels of abstraction; map divided into chunks and HPA* searches bt chunks
	"""

	counter = 0

	choice = randint(0, 4)  # allows pseudo random paths; choice used to modify cost calculation and move validity

	if choice == 4 and searchType == "HPA*":
		costMethod = 1
	else:
		costMethod = 0

	if choice < 1:
		checkOverlap = True
	else:
		checkOverlap = False

	searched = set()
	path = Path()

	targetFound = False

	frontier = PriorityQueue()  # uses priority queue rather than iterative approach to increase performance
	frontier.put((0, 0, self.cells[start].get()))

	cameFrom = {self.cells[start].get(): self.cells[start].get()}
	costSoFar = {self.cells[start].get(): 0}

	if target in self.walls or start == target:  # target can not be reached or is already reached
		path.fail()
		return path

	while not frontier.empty():
		current = frontier.get()[2]

		if current.location == target:
			targetFound = True
			break

		failed = 0

		for nextMove in [cellReference.get() for cellReference in current.neighbors[searchType]]:

			if searchType == "HPA*" and (not checkOverlap or
										 (nextMove.location in {start, target} or
										  not self.overlaps(nextMove.location, paths))) or \
					searchType == "A*" and self.validMove(current.location, nextMove.location):

				neighbor = self.cells[nextMove.location].get()

				if searchType == "A*":
					cost = costSoFar[current] + 1

				else:
					cost = costSoFar[current] + len(self.paths[(current.location, neighbor.location)].get())

				if neighbor not in costSoFar or cost < costSoFar[neighbor]:
					counter += 1

					costSoFar[neighbor] = cost

					priority = cost + self.h_cost(neighbor, target, costMethod)

					frontier.put((priority, counter, neighbor))  # counter acts as tiebreaker in case costs are same

					cameFrom[neighbor] = current

					searched.add(neighbor.location)

			else:
				failed += 1

		if failed == current.numNeighbors:  # returning bc no neighbors
			path.fail()
			return path

		if abort is not None and len(searched) >= abort:  # return bc has searched too many cells w/o finding path
			path.fail()
			return path

	if frontier.empty() and not targetFound:  # nothing valid left to search and target not reached
		altTargetFound = False

		if altTargets is not None:
			for pt in [cellReference.get().location for cellReference in altTargets]:
				if pt in searched:
					target = pt
					altTargetFound = True
					break

		if not altTargetFound:
			path.fail()

			if not checkOverlap and len(searched) > 0:
				path.trapped = True

			return path

	toAdd = self.cells[target].obj

	while toAdd != cameFrom[toAdd]:
		path.add(toAdd.location)
		toAdd = cameFrom[toAdd]

	path.add(self.cells[start].obj.location)
	path.reverse()

	if searchType == "A*":
		return path

	else:  # search type is HPA*; path found between chunks but still need to fill gaps with precomputed paths
		temp = Path()
		for i in range(len(path) - 1):
			for j in range(len(self.paths[(path[i], path[i + 1])].get()) - 1):
				temp.add(self.paths[(path[i], path[i + 1])].get()[j])
		temp.add(path.end)
		return temp