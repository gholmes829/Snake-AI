from core.searching import pathfind, hamiltonian

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
test = {
	(0, 0): 0,
	(1, 0): 0,
	(2, 0): 0,
	(3, 0): 0,
	(0, 1): 0,
	(1, 1): -1,
	(2, 1): 0,
	(3, 1): 0,
	(0, 2): 0,
	(1, 2): 0,
	(2, 2): 0,
	(3, 2): 0,
	(0, 3): -1,
	(1, 3): 0,
	(2, 3): 0,
	(3, 3): 0,
}

test[starting] = -1

for y in range(4):
	for x in range(4):
		print(test[(x, y)] + 1, end="")
	print()
print()
result = hamiltonian(test, (2, 0))
if result:
	path = result
	print("FOUND PATH", path)
	for x in range(4):
		for y in range(4):
			if test[(x, y)] == -1:
				test[(x, y)] = "x"
			else:
				test[(x, y)] = "-"
	
	for i, coord in enumerate(path):
		test[coord] = i%10
	
	for y in range(4):
		for x in range(4):
			print(test[(x, y)], end="")
		print()
