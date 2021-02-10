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

    def __call__(self, body, direction) -> tuple:
        """Does nothing, expandable"""
        raise NotImplementedError
        
    def getBrain(self):
        return {"type": "behavior"}
        
    def takeStock(self, body, direction, awareness, environment):
        return
        
    def reset(self):
        return
		
    @staticmethod
    def getMove(basis: tuple, direction: tuple) -> str:
        """
        Determines direction change from basis.

        Parameters
        ----------
        basis: tuple
            Original direction
        direction: tuple
            New direction after movement occurred

        Returns
        -------
        str: movement to go from basis to direction
        """
        return {(basis[1], -basis[0]): "left", basis: "straight", (-basis[1], basis[0]): "right"}[direction]

    @staticmethod
    def rotateCW(v: tuple) -> tuple:
        """
        Rotates input 90 degrees clockwise.

        Parameters
        ----------
        v: tuple
            Input vector to be rotated

        Returns
        -------
        tuple: v rotated by 90 degrees clockwise
        """
        return tuple(np.asarray(v, dtype=float) @ np.array([[0., 1.], [-1., 0.]]))

    @staticmethod
    def rotateCCW(v: tuple) -> tuple:
        """
        Rotates input 90 degrees counter-clockwise.

        Parameters
        ----------
        v: tuple
            Input vector to be rotated

        Returns
        -------
        tuple: v rotated by 90 degrees counter-clockwise
        """
        return tuple(np.asarray(v, dtype=float) @ np.array([[0., -1.], [1., 0.]]))
    
class AI(Behavior):
    def __init__(self):
        Behavior.__init__(self)

    @staticmethod
    def smartShield(decision: int, newDirection: int, move: str, vision: np.ndarray, direction):
        """
        Tries to prevent snake from going in dangerous direction.
        
        Parameters
        ----------
        decision: int
            Local direction
        newDirection: int
            Global direction
        move: string
            Move necessary to have Snake oriented to this direction
        vision: np.array
            Describes closeness of Snake's head to food, body, and wall
        direction: tuple
            Current global direction Snake is facing

        Returns
        -------
        tuple: (new global direction, move necessary to have Snake oriented to this direction)
        """
        possibleMoves = {0, 1, 2}
        possibleMoves.remove(decision)

        while possibleMoves and \
            ((decision == 0 and (vision[11] == 1 or vision[19] == 1)) or \
            (decision == 1 and (vision[8] == 1 or vision[16] == 1)) or \
            (decision == 2 and (vision[9] == 1 or vision[17] == 1))):
            decision = possibleMoves.pop()
            newDirection = {0: Behavior.rotateCCW(direction), 1: direction, 2: Behavior.rotateCW(direction)}[decision]
            move = {0: "left", 1: "straight", 2: "right"}[decision]

        return newDirection, move
        
# NON AI BEHAVIORS

class Manual(Behavior):
    """Provides direction based on keyboard input."""
    def __init__(self) -> None:
        """Initializes base class."""
        Behavior.__init__(self)

    def __call__(self, body, direction: tuple) -> tuple:
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
            move = direction

        newDirection = {False: move, True: direction}[move == (-direction[0], -direction[1])]
        move = Behavior.getMove(direction, newDirection)

        return newDirection, move

		
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

    def __call__(self, body, direction) -> tuple:
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
        newDirection = self.memories[self.t]
        move = Behavior.getMove(direction, newDirection)
        self.t += 1
        return newDirection, move

		
# AI BEHAVIORS

class NeuralNetwork(AI):
    """
    Uses neural network to decide direction.

    Attributes
    ----------
    confidence: list
        Log of 'confidence' of decisions
    """
    def __init__(self, layers=(24, 16, 3), smartShield: bool = False, **kwargs) -> None:
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
        AI.__init__(self)
        self.network = neural_nets.FFNN(layers, **kwargs)
        self.smartShield = smartShield
        self.vision = None

    def getBrain(self):
        return {"type": "neural network", "weights": self.network.weights, "biases": self.network.biases}

    def takeStock(self, body, direction, awareness, environment):
        self.vision, visionBounds = searching.castRays(body[0], direction, environment, awareness["maxVision"])
        return {"visionBounds": visionBounds}
        
    def __call__(self, body, direction) -> tuple:
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
        out = self.network.feedForward(self.vision)
        
        decision = np.argmax(out)
        newDirection = {0: Behavior.rotateCCW(direction), 1: direction, 2: Behavior.rotateCW(direction)}[decision]
        move = {0: "left", 1: "straight", 2: "right"}[decision] 
          
        if self.smartShield:
            newDirection, move = AI.smartShield(decision, newDirection, move, self.vision, direction)
          
        return newDirection, move
		
class Pathfinder(AI):
    def __init__(self):
        AI.__init__(self)
        self.path = []
		
    def takeStock(self, body, direction, awareness, environment):
        if not self.path:
            self.path = searching.pathfind(environment, body[0], environment.filter(1)[0])[:-1]
        return {"path": set(self.path)}

    def __call__(self, body, direction):
        if self.path:
            moveTo = self.path.pop()
            newDirection = (moveTo[0] - body[0][0], moveTo[1] - body[0][1])
            move = {Behavior.rotateCCW(direction): "left", direction: "straight", Behavior.rotateCW(direction): "right"}[newDirection]
        else:
            newDirection, move = direction, "straight"

        return newDirection, move
		
		
class Hamiltonian(AI):
    def __init__(self):
        AI.__init__(self)
        self.path = []
		
    def takeStock(self, body, direction, awareness, environment):
        if not self.path:
            #print()
            #print(body)
            copy = deepcopy(environment)
            copy[body[-1]] = 0
            if (initialPath := searching.longestPath(copy, body[0], body[-1], environment.filter(1)[0])[:-1]):
                connection = body[:-1]
                #print("Longest path", initialPath)
                #print("Connection", connection)
                #print("Final path:", connection + initialPath)
                self.path = connection + initialPath
            #if self.path:
                #print(body[0])
                #print("Cycle found!", self.path)
        return {"path": set(self.path)}

    def __call__(self, body, direction):

        if self.path:
            moveTo = self.path.pop()
            #print(body[0], moveTo)
            newDirection = (moveTo[0] - body[0][0], moveTo[1] - body[0][1])
            move = {Behavior.rotateCCW(direction): "left", direction: "straight", Behavior.rotateCW(direction): "right"}[newDirection]
        else:
            newDirection, move = direction, "straight"
            print("GOING STRAIGHT!!!!!!!!!!!!!")
        return newDirection, move
		
		
class FloodPathfinder(AI):
    def __init__(self):
        AI.__init__(self)
        self.path = []
        self.open = {(-1, 0): 0, (0, -1): 0, (1, 0): 0}
		
    def takeStock(self, body, direction, awareness, environment):
        if not self.path:
            self.path = searching.pathfind(environment, body[0], environment.filter(1)[0])[:-1]
			
        moves = {(-1, 0): Behavior.rotateCCW(direction), (0, -1): direction, (1, 0): Behavior.rotateCW(direction)}
        for turnDirection in self.open:
            newDirection = moves[turnDirection]
            if (newPos := (body[0][0] + newDirection[0], body[0][1] + newDirection[1])) in environment and environment[newPos] != -1:
    
                self.open[turnDirection] = searching.floodFillCount(deepcopy(environment), newPos)
            else:
                self.open[turnDirection] = 0
			
        return {"path": set(self.path)}

    def __call__(self, body, direction):
        if self.path:
            moveTo = self.path.pop()
            newDirection = (moveTo[0] - body[0][0], moveTo[1] - body[0][1])
            move = {Behavior.rotateCCW(direction): "left", direction: "straight", Behavior.rotateCW(direction): "right"}[newDirection]
        else:
            if sum(self.open.values()) != 0:
                moves = {(-1, 0): Behavior.rotateCCW(direction), (0, -1): direction, (1, 0): Behavior.rotateCW(direction)}
                turn = max(self.open, key=self.open.get)
                newDirection = moves[turn]
                move = {Behavior.rotateCCW(direction): "left", direction: "straight", Behavior.rotateCW(direction): "right"}[newDirection]
            else:
                newDirection, move = direction, "straight"

        return newDirection, move
		
class FloodFill(AI):
    def __init__(self):
        AI.__init__(self)
        self.open = {(-1, 0): 0, (0, -1): 0, (1, 0): 0}
		
    def takeStock(self, body, direction, awareness, environment):
        moves = {(-1, 0): Behavior.rotateCCW(direction), (0, -1): direction, (1, 0): Behavior.rotateCW(direction)}
        for turnDirection in self.open:
            newDirection = moves[turnDirection]
            if environment[(newPos := (body[0][0] + newDirection[0], body[0][1] + newDirection[1]))] != -1:
    
                self.open[turnDirection] = searching.floodFillCount(deepcopy(environment), newPos)
            else:
                self.open[turnDirection] = 0
        #self.open = {turn:searching.floodFillCount(environment, ) for turn in self.open}
        return {"open": self.open}

    def __call__(self, body, direction):
        if sum(self.open.values()) != 0:
            moves = {(-1, 0): Behavior.rotateCCW(direction), (0, -1): direction, (1, 0): Behavior.rotateCW(direction)}
            turn = max(self.open, key=self.open.get)
            newDirection = moves[turn]
            move = {Behavior.rotateCCW(direction): "left", direction: "straight", Behavior.rotateCW(direction): "right"}[newDirection]
        else:
            newDirection, move = direction, "straight"

        return newDirection, move