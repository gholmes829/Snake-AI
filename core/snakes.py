"""
Snake objects that can play Snake game in an environment.

Classes
-------
Snake
    Snake base class.
SmartSnake
    Snake predetermined with neural net as behavior.
Player
    Snake predetermined with Manual player controller.
Ghost
    Snake with predetermined queue of moves.
"""

from random import choice
from numba import jit
import weakref
import numpy as np

from core.constants import *
from core import behaviors

__author__ = "Grant Holmes"
__email__ = "g.holmes429@gmail.com"


class _SnakeBase:
    """
    Snake base class.

    Attributes
        ----------
        initialSize: int
            Initial size of Snake's body
        vision: int
            Max number of steps Snake can see in
        score: int
            Number of times Snake gets food
        hunger: int
            How hungry Snake is, increases by 1 every time Snake moves, decreases by refeed when Snake eats
        age: int
            How old Snake is, increases by 1 every time Snake moves
        dead: bool
            If snake is dead or not
        body: list
            (x, y) coordinates of Snake's body
        head: tuple
            (x, y) coordinate of Snake's head
        direction: tuple
            (x, y) direction Snake is currently moving in
        prevTail: tuple
            (x, y) coordinate of previous pos of Snake's tail
        moveCount: dict
            Number of times Snake has moved in each direction
        kwargs: dict
            Saved copy of key word arguments so Snake can reset

    Public Methods
    --------------
    move(vision: list) -> None:
        Snake picks a direction to move and moves one step that direction.
    getBrain() -> Dict:
        Gets weights and biases of neural network if Snake controlled by AI.
    kill() -> None:
        Kills snake.
    revive() -> None:
        Resets Snake to allow Snake to play multiple games.
    grow() -> None:
        Increases body size by one, grows segment where tail used to be.
    setReference(origin: tuple) -> None:
        Moves Snake's coordinate frame of reference to new origin point.
    """
    def __init__(self,
				 behavior: behaviors.Behavior,
                 initialSize: int = 4,
                 vision: int = 10,
                 starvation: dict = {"active": True, "maxHunger": lambda size: 200, "refeed": lambda size: 150},
                 color: tuple = None) -> None:
        """
        Initializes.

        Parameters
        ----------
        behavior: behaviors.behavior
            Called to determine Snake's next move based on Snake's vision
        starvation: bool, default=True
            Indicates whether Snake can starve to death or not
        initialSize: int, default=4
            Initial size of Snake's body
        vision: int, default=10
            Max number of steps Snake can see in
        maxHungerFunc: callable, default=lambda size: 200.
            Max hunger Snake can have before it dies
        refeedFunc: callable, default=lambda size: 150.
            How much Snake's hunger is reduces when it eats food
        color: tuple, optional
            Determines hunger of snake, color is random if not passed in
        """
        self.behavior = behavior
        self.maxVision = vision
		
        self.kwargs = {
            "starvation": starvation,
            "initialSize": initialSize,
            "vision": vision,
            "color": color
        }
		
        self.vision = None
        self.visionBounds = []

        self.starvation = starvation["active"]
        self.initialSize = initialSize
 
        self.maxHungerFunc = starvation["maxHunger"]
        self.maxHunger = self.maxHungerFunc (initialSize)
        self.refeedFunc = starvation["refeed"]
        self.refeed = self.refeedFunc(initialSize)

        self.color = choice(SNAKE_COLORS) if color is None else color
        self.score, self.hunger, self.age = 0, 0, 0
        self.dead = False

        self.body = []
        self.head = (0, 0)
        self.direction = (1, 0)

        self.body.append((0, 0))
        for i in range(self.initialSize - 1):
            self.body.append((-1 * (i + 1), 0))
        self.prevTail = (-1 * self.initialSize, 0)

        self.moveCount = {"left": 0, "straight": 0, "right": 0}  # number of times Snake has moved in each direction

    def move(self) -> None:
        """
        Snake picks a direction to move and moves one step that direction.

        Parameters
        ----------
        vision: list
            24x1 list of floats, 0-7 closeness to food, 8-15 closeness to body, 16-23 closeness to wall
        """
        self.prevTail = self.body[-1]
        self.direction, move = self.behavior(*self.getState())
        self.moveCount[move] += 1

        self.body.pop()
        self.body.insert(0, (self.body[0][0] + self.direction[0], self.body[0][1] + self.direction[1]))
        self.head = self.body[0]

        if self.head in self.body[1:] or (self.starvation and self.hunger >= self.maxHunger):
            self.kill()
        else:
            self.hunger += 1
            self.age += 1

    def revive(self) -> None:
        """Resets Snake to allow Snake to play multiple games."""
        _SnakeBase.__init__(self, self.behavior, **self.kwargs)

    def kill(self) -> None:
        """Kills snake."""
        self.dead = True
     
    def updateVision(self, environment) -> None:
        self.vision = self._castRays(environment)

    def grow(self) -> None:
        """Increases body size by one, grows segment where tail used to be."""
        self.score += 1
        self.body.append(self.prevTail)
        self.prevTail = (2 * self.prevTail[0] - self.body[-2][0], 2 * self.prevTail[1] - self.body[-2][1])
        self.maxHunger = self.maxHungerFunc(len(self))
        self.refeed = self.refeedFunc(len(self))
        self.hunger = (self.hunger - self.refeed) * ((self.hunger - self.refeed) > 0)

    def setReference(self, origin: tuple) -> None:
        """
        Moves Snake's coordinate frame of reference to new origin point.

        Parameters
        ----------
        origin: tuple
            Origin for new frame of reference
        """
        self.body = [(origin[0] - self.head[0] + segment[0], origin[1] - self.head[1] + segment[1]) for segment in
                     self.body]
        self.head = origin
		
    def _castRays(self, surroundings) -> list:
        """
        Cast octilinear rays out from Snake's head to provide Snake awareness of its surroundings.

        Note
        ----
        'Closeness' defined as 1/dist.
        """
        origin = self.head
        snakeDirection = self.direction
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
            rays[direction] = {"wall": 1 / distance * int(distance <= self.maxVision), "food": 0, "body": 0}

        self.visionBounds.clear()  # reset so rays contains info only of this instance
        probe = None
        for ray, targets in rays.items():  # ...in each 8 octilinear directions
            bound = min(limits[ray], self.maxVision)
            step = 1
            while not targets["food"] and not targets["body"] and step <= bound:  # take specified number of steps away from Snake's head and don't let rays search outside of map borders
                probe = (origin[0] + ray[0] * step, origin[1] + ray[1] * step)  # update probe position
                if not targets["food"] and surroundings[probe] == FOOD:  # if food not found yet and found food
                    targets["food"] = 1 / _SnakeBase.dist(origin, probe)
                elif not targets["body"] and surroundings[probe] == DANGER:  # if body not found yet and found body
                    targets["body"] = 1 / _SnakeBase.dist(origin, probe)
                step += 1	
            self.visionBounds.append((origin, (origin[0] + ray[0] * bound, origin[1] + ray[1] * bound)))  # add end of ray to list

        data = np.zeros(24)

        for i, direction in enumerate(DIRECTIONS):  # for each direction
            for j, item in ((0, "food"), (8, "body"), (16, "wall")):
                # need to change reference so 'global up' will be 'Snake's left' is Snake if facing 'global right'
                data[i + j] = rays[_SnakeBase.getLocalDirection(snakeDirection, direction)][item]  # add data

        # PRINT VALUES OF DATA TO DEBUG
        #for i in range(3):
        #    for j in range(8):
        #        print(round(data[i * 8 + j], 3), end=" ")
        #    print()
        #self.display()
        return data

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
		
    def getState(self, environment: dict) -> any:
        raise NotImplementedError
        
    def __len__(self):
        """Returns length of snake's body"""
        return len(self.body)

		
class Player(_SnakeBase):
    """Snake predetermined with Manual player controller."""
    def __init__(self, **kwargs) -> None:
        """
        Initializes with Manual player controller.

        Parameters
        ----------
        **kwargs:
            parameters for base class
        """
        _SnakeBase.__init__(self, behaviors.Manual(), **kwargs)
	
    def getState(self, environment) -> list:
        return [self.direction]

		
class Ghost(_SnakeBase):
    """Snake with predetermined queue of moves."""
    def __init__(self, memories: list, **kwargs) -> None:
        """
        Initializes with Manual player controller.

        Parameters
        ----------
        memories: list
            list of moves ordered with time
        **kwargs:
            parameters for base class
        """
        _SnakeBase.__init__(self, behaviors.Replay(memories), starvation=True, **kwargs)
		
    def getState(self, environment):
        return [self.direction]

class BoundMethod:
    def __init__(self, instance, func):
        self.func = func
        self.instance_ref = weakref.ref(instance)

        self.__wrapped__ = func

    def __call__(self, *args, **kwargs):
        instance = self.instance_ref()
        return self.func(instance, *args, **kwargs)


class AI(_SnakeBase):
    """Snake predetermined with neural net as behavior."""
    _behaviors = {
        "neural net": lambda args, kwargs: behaviors.NeuralNetwork(*args, **kwargs),
    }
	
    _behaviorMethods = {
        "neural net": {
            "getState": lambda self: [self.vision, self.direction],
            "getSensoryInputs": lambda self, environment: self.updateVision(environment),
            "getBrain": lambda self: self.behavior.getNetwork
        },
    }

    def __init__(self, behaviorType, behaviorArgs: list = None, behaviorKwargs: dict = None, **kwargs: dict) -> None:
        """
        Initializes with loaded model or creates random model

        Parameters
        ----------
        model: dict, optional
            {"weights": neural net weight, "biases": neural net biases}, load neural net
        layers: tuple, default=(24, 16, 3)
            Layers architecture of neural network
        smartShield: bool, default=False
            Determines whether dangerous moves can be overwritten

        **kwargs:
            Parameters for base class
        """
        if behaviorArgs is None:
            behaviorArgs = []
            
        if behaviorKwargs is None:
            behaviorKwargs = {}
        behavior = AI._behaviors[behaviorType](behaviorArgs, behaviorKwargs)
        for name, method in AI._behaviorMethods[behaviorType].items():
            self.__dict__[name] = BoundMethod(self, method)
        _SnakeBase.__init__(self, behavior, **kwargs)
		
    def getBrain(self) -> dict:
        """
        Gets weights and biases of neural network if Snake controlled by AI.

        Returns
        -------
        dict: dict("weights": neural net weights, "biases": neural net biases)

        Raises
        ------
        TypeError: behavior doesn't have weights and biases if not AI based
        """
        return self.behavior.getNetwork()

    @staticmethod
    def fitness(snake) -> None:
        """
        ((snake_score)^3 * snake_age)/1000 + 1 if moved in all directions else 0
        """
        return ((snake.score ** 3) * snake.age) / 1000 + 1 if all([p > 0 for p in snake.moveCount.values()]) else 0

    @staticmethod
    def mergeTraits(child: object, parent1: object, parent2: object) -> None:
        """
        Defines how Snake's traits are combined during genetic crossover.

        child: Snake
            Snake inheriting traits from parents
        parent1: Snake
            Parent whose traits are passed down to child
        parent2: Snake
            Parent whose traits are passed down to child
        """
        child.color = ((parent1.color[0] + parent2.color[0]) / 2, (parent1.color[1] + parent2.color[1]) / 2, (parent1.color[2] + parent2.color[2]) / 2)
