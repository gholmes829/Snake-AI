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

from core.constants import *
from core import behaviors

__author__ = "Grant Holmes"
__email__ = "g.holmes429@gmail.com"


class Snake:
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
    translate(origin: tuple) -> None:
        Moves Snake's coordinate frame of reference to new origin point.
    """
    def __init__(self, behavior: behaviors.Behavior,
                 starvation: bool = True,
                 initialSize: int = 4,
                 vision: int = 10,
                 maxHunger: float = 200,
                 refeed: float = 150,
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
        maxHunger: float, default=200.
            Max hunger Snake can have before it dies
        refeed: float, default=150.
            How much Snake's hunger is reduces when it eats food
        color: tuple, optional
            Determines hunger of snake, color is random if not passed in
        """
        self.behavior = behavior

        self.starvation = starvation
        self.initialSize = initialSize
        self.vision = vision
        self.maxHunger = maxHunger
        self.refeed = refeed

        self.kwargs = {
            "starvation": starvation,
            "initialSize": initialSize,
            "vision": vision,
            "maxHunger": maxHunger,
            "refeed": refeed,
            "color": color
        }

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

    def move(self, vision: list) -> None:
        """
        Snake picks a direction to move and moves one step that direction.

        Parameters
        ----------
        vision: list
            24x1 list of floats, 0-7 closeness to food, 8-15 closeness to body, 16-23 closeness to wall
        """
        self.age += 1

        self.prevTail = self.body[-1]
        self.direction, move = self.behavior(vision, self.direction)
        self.moveCount[move] += 1

        self.body.pop()
        self.body.insert(0, (self.body[0][0] + self.direction[0], self.body[0][1] + self.direction[1]))
        self.head = self.body[0]

        if self.head in self.body[1:] or (self.starvation and self.hunger > self.maxHunger):
            self.kill()
        else:
            self.hunger += 1

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
        if type(self.behavior) == behaviors.AI:
            return {"weights": self.behavior.weights, "biases": self.behavior.biases}
        else:
            raise TypeError("Behavior is not neural net based")

    def kill(self) -> None:
        """Kills snake."""
        self.dead = True

    def revive(self) -> None:
        """Resets Snake to allow Snake to play multiple games."""
        Snake.__init__(self, self.behavior, **self.kwargs)

    def grow(self) -> None:
        """Increases body size by one, grows segment where tail used to be."""
        self.body.append(self.prevTail)
        self.prevTail = (2 * self.prevTail[0] - self.body[-2][0], 2 * self.prevTail[1] - self.body[-2][1])
        self.hunger = (self.hunger - self.refeed) * (self.hunger - self.refeed >= 0)
        self.score += 1

    def translate(self, origin: tuple) -> None:
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
        merged = []
        for i in range(3):
            mergeType = choice(["swap", "average"])
            if mergeType == "swap":
                merged.append(choice((parent1.color[i], parent2.color[i])))
            else:
                merged.append((parent1.color[i] + parent2.color[i]) / 2)
        child.color = tuple(merged)


class SmartSnake(Snake):
    """Snake predetermined with neural net as behavior."""
    def __init__(self, model: dict = None, layers: tuple = (24, 16, 3), smartShield: bool = False, **kwargs) -> None:
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
        self.model = model
        self.layers = layers
        if model is None:
            Snake.__init__(self, behaviors.AI(layers=layers, smartShield=smartShield), starvation=True, **kwargs)
        else:
            Snake.__init__(self, behaviors.AI(layers=layers, smartShield=smartShield, **model), starvation=True, **kwargs)


class Player(Snake):
    """Snake predetermined with Manual player controller."""
    def __init__(self, **kwargs) -> None:
        """
        Initializes with Manual player controller.

        Parameters
        ----------
        **kwargs:
            parameters for base class
        """
        Snake.__init__(self, behaviors.Manual(), starvation=False, **kwargs)


class Ghost(Snake):
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
        Snake.__init__(self, behaviors.Replay(memories), starvation=True, **kwargs)
