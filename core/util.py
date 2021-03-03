"""

"""
from core.constants import *

from math import sqrt
import numpy as np

def getValidInput(msg: str,
				  dtype: any = str,
				  lower: float = None, upper: float = None,
				  valid: set = None,
				  isValid: callable = None) -> any:
	"""
	Gets input from user constrained by parameters.

	Parameters
	----------
	msg: str
		Message to print out to user requesting input
	dtype: any, default=str
		Type that input will get converted to
	lower: float, optional
		Numerical lower bound
	upper: float, optional
		Numerical upper bound
	valid: set, optional
		Set of possible valid inputs
	isValid: callable, optional
		Function returning bool to determine if input is valid

	Returns
	-------
	any: valid user input
	"""
	print(msg)
	while True:
		try:
			choice = dtype(input("\nChoice: "))
		except ValueError:  # if type can't be properly converted into dtype
			continue
		if (lower is None or choice >= lower) and \
				(upper is None or choice <= upper) and \
				(valid is None or choice in valid) and \
				(isValid is None or isValid(choice)):
			return choice
			
def getSelection(*args, msg: str = "Choose item:") -> tuple:
	for i, item in enumerate(args):
		msg += "\n\t" + str(i + 1) + ") " + str(item)
		
	i = getValidInput(msg, dtype=int, lower=1, upper=len(args)) - 1
	return i, args[i]
	
def loadNPZ(path):
	 return np.load(path, allow_pickle=True)
	 
cos45 = sqrt(2)/2
	 
# CONVERT TO GENERAL ANGLE FUNCTION
	 
def rotate90CCW(v):
	return (-v[1], v[0])

def rotate90CW(v):
	return (v[1], -v[0])
	
def rotate45CCW(v):
	return (round(cos45*(v[0] - v[1]), 0), round(cos45*(v[0]+v[1]), 0))

def rotate45CW(v):
	for _ in range(7):
		v = rotate45CCW(v)
	return v
	
def rotate135CCW(v):
	for _ in range(3):
		v = rotate45CCW(v)
	return v

def rotate135CW(v):
	for _ in range(5):
		v = rotate45CCW(v)
	return v
	
def rotate180(v):
	return (-v[0], -v[1])
	
#  localGlobal -- know prev direction and turn about to make, want to know global direction
#  globalLocal --  # know prev direction and new global direction, want to know what turn you took to get to new global direction

orientedDirections = {}
tempLocal = lambda currDirection: {(-1, 0): rotate90CCW(currDirection), (0, 1): currDirection, (1, 0): rotate90CW(currDirection), (0, -1): rotate180(currDirection)}

for curr in ORTHOGONAL:
	for new in ORTHOGONAL:
		orientedDirections["local", curr, new] = tempLocal(curr)[new]
		
for curr in ORTHOGONAL:
	possibleNew = {
		rotate90CCW(curr): (-1, 0),
		curr: (0, 1),
		rotate90CW(curr): (1, 0),
		rotate180(curr): (0, -1),
		rotate45CW(curr): (1, 1),
		rotate45CCW(curr): (-1, 1),
		rotate135CW(curr): (1, -1),
		rotate135CCW(curr): (-1, -1),
	}
	for new in DIRECTIONS:
		orientedDirections["global", curr, new] = possibleNew[new]		
	
def getOrientedDirection(currDirection, newDirection, directionType) -> tuple:
	return orientedDirections[directionType, currDirection, newDirection]
	
localizedDirections = {}
	
for curr in ORTHOGONAL:
	for new in DIRECTIONS:
		localizedDirections[curr, new] = getOrientedDirection(curr, new, "global")
	
[print(key, "|", value, "\n") for key, value in orientedDirections.items()] 	
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
	
