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

from core import neural_nets
from core.constants import *

__author__ = "Grant Holmes"
__email__ = "g.holmes429@gmail.com"


class Behavior:
    """Interface class with helper functions."""
    def __init__(self) -> None:
        """Does nothing, expandable"""
        pass

    def __call__(self) -> tuple:
        """Does nothing, expandable"""
        raise NotImplementedError
    
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

    def __call__(self, vision: np.ndarray, direction: tuple) -> tuple:
        """
        Returns keyboard input, ignores vision.

        Parameters
        ----------
        vision: np.ndarray
            Describes closeness of Snake's head to food, body, and wall
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


class NeuralNet(neural_nets.FFNN, Behavior):
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
        neural_nets.FFNN.__init__(self, layers, **kwargs)
        self.smartShield = smartShield

    def getNetwork(self):
        return {"weights": self.weights, "biases": self.biases}
		
    def __call__(self, vision: np.ndarray, direction: tuple) -> tuple:
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
        out = self.feedForward(vision)
        
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
    def __init__(self, data: list) -> None:
        """
        Initializes base class.

        Parameters
        ---------
        data: list
            List of (x, y) moves
        """
        Behavior.__init__(self)
        self.data = data
        self.t = 0

    def __call__(self, vision: np.ndarray, direction: tuple) -> tuple:
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
        newDirection = self.data[self.t]
        move = Behavior.getMove(direction, newDirection)
        self.t += 1
        return newDirection, move
