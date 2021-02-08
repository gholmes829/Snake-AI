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
from numba import jit

from core import neural_nets, pathfinding
from core.constants import *

__author__ = "Grant Holmes"
__email__ = "g.holmes429@gmail.com"

# BEHAVIOR FACTORY
def getBehavior(behaviorType, *args, **kwargs):
    return {
        "neural network": NeuralNetwork,
        "ghost": Replay,
        "player": Manual,
    }[behaviorType](*args, **kwargs)


class Behavior:
    """Interface class with helper functions."""
    def __init__(self) -> None:
        """Does nothing, expandable"""
        if type(self) == Behavior:
            raise NotImplementedError

    def __call__(self, awareness) -> tuple:
        """Does nothing, expandable"""
        raise NotImplementedError
        
    def getBrain(self):
        return {"type": "behavior"}
        
    def takeStock(self, head, direction, awareness, surroundings):
        return
        
    def reset(self):
        return
    
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


    @staticmethod    
    def castRays(origin, orientation, surroundings, maxRaySize) -> list:
        """
        Cast octilinear rays out from Snake's head to provide Snake awareness of its surroundings.

        Note
        ----
        'Closeness' defined as 1/dist.
        """
        limits, rays = {}, {}

        # get distance from Snake's head to map borders
        bounds = {
            UP: origin[1],
            RIGHT: (surroundings.size[0] - origin[0] - 1),
            DOWN: (surroundings.size[1] - origin[1] - 1),
            LEFT: origin[0]
        }

        # determine how far rays can go
        for direction in ORTHOGONAL:
            limits[direction] = bounds[direction]

        for diagonal in DIAGONAL:
            limits[diagonal] = min(limits[(diagonal[0], 0)], limits[(0, diagonal[1])])

        # determine closeness of Snake to walls, initialize rays dict
        for direction in DIRECTIONS:
            distance = limits[direction] + 1 if direction in ORTHOGONAL else (limits[direction] + 1) * 1.414
            rays[direction] = {"wall": 1 / distance * int(distance <= maxRaySize), "food": 0, "body": 0}

        visionBounds = []
        probe = None
        for ray, targets in rays.items():  # ...in each 8 octilinear directions
            bound = min(limits[ray], maxRaySize)
            step = 1
            while not targets["food"] and not targets["body"] and step <= bound:  # take specified number of steps away from Snake's head and don't let rays search outside of map borders
                probe = (origin[0] + ray[0] * step, origin[1] + ray[1] * step)  # update probe position
                if not targets["food"] and surroundings[probe] == FOOD:  # if food not found yet and found food
                    targets["food"] = 1 / Behavior.dist(origin, probe)
                elif not targets["body"] and surroundings[probe] == DANGER:  # if body not found yet and found body
                    targets["body"] = 1 / Behavior.dist(origin, probe)
                step += 1	
            visionBounds.append((origin, (origin[0] + ray[0] * bound, origin[1] + ray[1] * bound)))  # add end of ray to list

        vision = np.zeros(24)

        for i, direction in enumerate(DIRECTIONS):  # for each direction
            for j, item in ((0, "food"), (8, "body"), (16, "wall")):
                # need to change reference so 'global up' will be 'Snake's left' is Snake if facing 'global right'
                vision[i + j] = rays[Behavior.getLocalDirection(orientation, direction)][item]  # add data

        # PRINT VALUES OF DATA TO DEBUG
        #for i in range(3):
        #    for j in range(8):
        #        print(round(data[i * 8 + j], 3), end=" ")
        #    print()
        return vision, visionBounds

    @staticmethod
    def getLocalDirection(basis: tuple, direction: tuple) -> tuple:
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

    @staticmethod
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
        

class Manual(Behavior):
    """Provides direction based on keyboard input."""
    def __init__(self) -> None:
        """Initializes base class."""
        Behavior.__init__(self)

    def __call__(self, head, direction: tuple, awareness) -> tuple:
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


class NeuralNetwork(Behavior):
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
        Behavior.__init__(self)
        self.network = neural_nets.FFNN(layers, **kwargs)
        self.smartShield = smartShield

    def getBrain(self):
        return {"type": "neural network", "weights": self.network.weights, "biases": self.network.biases}

    def takeStock(self, head, direction, awareness, surroundings):
        vision, visionBounds = Behavior.castRays(head, direction, surroundings, awareness["maxVision"])
        awareness["vision"] = vision
        awareness["visionBounds"] = visionBounds
        
    def __call__(self, head, direction, awareness) -> tuple:
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
        vision = awareness["vision"]
        out = self.network.feedForward(vision)
        
        decision = np.argmax(out)
        newDirection = {0: Behavior.rotateCCW(direction), 1: direction, 2: Behavior.rotateCW(direction)}[decision]
        move = {0: "left", 1: "straight", 2: "right"}[decision] 
          
        if self.smartShield:
            newDirection, move = Behavior.smartShield(decision, newDirection, move, vision, direction)
          
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

    def __call__(self, head, direction, awareness) -> tuple:
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