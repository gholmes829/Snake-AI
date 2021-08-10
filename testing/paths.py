"""
Testing environment and unit tests in developments...
"""

from core.game.brain import pathfind, hamiltonian, longestPath

# PATHFINDING

print("PATHFINDING:")
print()

testInput = {}

for x in range(10):
	for y in range(10):
		testInput[(x, y)] = -1 if x%3 and y%4 else 0
		
#testInput[(9, 0)] = -1
#testInput[(6, 0)] = -1
#testInput[(3, 0)] = -1
#testInput[(1, 4)] = -1
#testInput[(0, 1)] = -1
#testInput[(1, 0)] = -1
#print(testInput[(0, 9)])

for y in range(10):
	for x in range(10):
		print({-1: "X", 0: "|"}[testInput[(x, y)]], end="")
	print()
	
print()
path = pathfind(testInput, (0, 0), (9, 9))

result = {}

for y in range(10):
	for x in range(10):
		result[(x, y)] = "-" if x%3 and y%4 else " "
		
for i, coord in enumerate(path):
	result[coord] = str(i%10)

for y in range(10):
	for x in range(10):
		print(result[(x, y)], end="")
	print()

print("\nPATHFINDING DONE!")





# HAMILTONIAN
print("\n\n")
print("HAMILTONIAN:")

starting = (2, 0)

n = 4
test = {}



for x in range(n):
	for y in range(n):
		test[(x, y)] = 0
#test[(0, 1)] = -1
test[starting] = -1

for y in range(n):
	for x in range(n):
		print(test[(x, y)] + 1, end="")
	print()
print()
result = hamiltonian(test, starting, starting)
if result:
	path = result
	print("FOUND PATH", path)
	for x in range(n):
		for y in range(n):
			if test[(x, y)] == -1:
				test[(x, y)] = "x"
			else:
				test[(x, y)] = "-"
	
	for i, coord in enumerate(path):
		test[coord] = i%10
	
	for y in range(n):
		for x in range(n):
			print(test[(x, y)], end="")
		print()

		
# LONGEST PATH

print("\nLONGEST PATH:")
print()

testInput = {}

n=6

for x in range(n):
	for y in range(n):
		testInput[(x, y)] = 0
	
testInput[(1, 1)] = -1
testInput[(2, 1)] = -1	
testInput[(3, 1)] = -1
testInput[(4, 1)] = -1
testInput[(1, 4)] = -1
testInput[(2, 4)] = -1	
testInput[(3, 4)] = -1
testInput[(4, 4)] = -1	
testInput[(1, 2)] = -1
testInput[(1, 3)] = -1	
testInput[(4, 2)] = -1
testInput[(4, 3)] = -1	

for y in range(n):
	for x in range(n):
		print({-1: "X", 0: "|"}[testInput[(x, y)]], end="")
	print()
	
print()
path = longestPath(testInput, (2, 1), (3, 1), (0, 2), depth=10)

result = {}

for y in range(n):
	for x in range(n):
		result[(x, y)] = "-" if testInput[(x, y)] == -1 else " "
		
for i, coord in enumerate(path):
	result[coord] = str(i%10)

for y in range(n):
	for x in range(n):
		print(result[(x, y)], end="")
	print()
	
for i in range(len(path) - 1):
	first = path[i]
	second = path[i+1]
	diff = (second[0]-first[0], second[1]-first[1])
	if diff not in {(1, 0), (-1, 0), (0, 1), (0, -1)}:
		print("Hole", i, first, second, diff)
		raise AssertionError("path has holes", path)

print()
print(path)

print("\nPATHFINDING DONE!")