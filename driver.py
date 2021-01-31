"""
Contains Driver class, controls flow of program.

Classes
-------
Driver
    Allows user to select modes. Controls flow of program.
"""

import os
import sys

import json
import numpy as np
from time import time

from core import genetics, game, environments, snakes, settings

__author__ = "Grant Holmes"
__email__ = "g.holmes429@gmail.com"


class Driver:
    """
    Allows user to select modes. Controls flow of program.

    Attributes
    ----------
    environment: Environment
        last environment used/ in use
    replayPath: str
        Path to replay folder
    dnaPath: str
        Path to dna folder
    modelPath: path to model folder

    Public Methods
    --------------
    run() -> None
        Allows user to select mode, runs mode
    """
    def __init__(self) -> None:
        """Gets paths to necessary folders."""
        self.environment = None
        self.replayPath = os.path.join(os.getcwd(), "replays")
        self.dnaPath = os.path.join(os.getcwd(), "dna")
        self.modelPath = os.path.join(os.getcwd(), "trained")
        print("    +" + "="*8 + "+")
        print("    |SNAKE AI|")
        print("    +" + "="*8 + "+")
        print("\nInitialized with the following settings:")
        print(settings.getInfo(), "\n")

    def run(self) -> None:
        """Allows user to select mode, runs mode."""
        mode = 0
        modes = {
            1: self._playClassic,
            2: self._playAI,
            3: self._trainAI,
            4: self._watchReplay,
            5: self._watchSaved,
            6: Driver._exit
        }
        msg = "Select mode:"
        msg += "\n\t1) Play Classic\n\t2) Play AI\n\t3) Train AI\n\t4) Replay Last Game\n\t5) Watch Saved\n\t6) Exit"

        while mode != 6:
            mode = Driver.getValidInput(msg, dtype=int, lower=1, upper=len(modes))
            modes[mode]()
            print()

    def _playClassic(self) -> None:
        """Opens GUI window and lets user play Snake with keyboard controls."""
        snake = snakes.Player(**settings.snakeParams)
        self.environment = environments.Environment(snake, settings.mapSize)
        game.playGame(self.environment)
        print()
        self._checkSave()

    def _playAI(self) -> None:
        """User selects saved model from .../dna/trained. Opens GUI window and AI plays game."""
        trainedFiles = os.listdir(self.modelPath)
        numTrained = len(trainedFiles)
        if numTrained == 0:
            print("No trained AI!\n")
        else:
            msg = "Select AI to use:"
            for i, model in enumerate(trainedFiles, start=1):
                msg += "\n\t" + str(i) + ") " + str(model)
            index = Driver.getValidInput(msg, dtype=int, valid=range(1, numTrained + 1)) - 1
            modelFile = trainedFiles[index]

            data = np.load(os.path.join(self.modelPath, modelFile), allow_pickle=True)

            model = {
                "weights": [np.asarray(layer, dtype=float) for layer in data["weights"]],
                "biases": [np.asarray(layer, dtype=float) for layer in data["biases"]]
            }

            dataParams = {
                "model": model,
                "layers": settings.networkArchitecture,
                "color": tuple(data["color"]),
                "smartShield": settings.smartShield
            }

            snake = snakes.SmartSnake(**settings.snakeParams, **dataParams)
            self.environment = environments.Environment(snake, settings.mapSize)
            game.playGame(self.environment, render=True)
            self._checkSave()

    def _checkSave(self) -> None:
        """Checks to see if user wants to save last game to .../replays. If so, user inputs file name."""
        choice = Driver.getValidInput("Save game as replay?\n\t1) Yes\n\t2) No", dtype=int, valid={1, 2})
        shouldSave = {1: True, 2: False}[choice]
        if shouldSave:
            print()
            self._saveGame(Driver.getValidInput("File name?"))

    def _saveGame(self, name: str) -> None:
        """
        Saves game environment to .json file in .../replay folder.

        Parameters
        ----------
        name: str
            File name for saved game file
        """
        if not Driver.hasExt(name, ".json"):
            name += ".json"
        data = self.environment.getData()
        with open(os.path.join(self.replayPath, name), "w") as f:
            json.dump(data, f, indent=4)

    def _trainAI(self) -> None:
        """Trains AI snakes. Saves data and models in ../dna folder. Creates new folder for each training session."""
        # settings validation
        if settings.populationSize < 5:
            print("\nError: Population size must be at least 5. Change size in settings.py.")
            return
            
        # initialize paths and files
        dnaFiles = os.listdir(self.dnaPath)
        if len(dnaFiles) > 0:
            currEvolution = max([int(file[file.index("_")+1:]) for file in dnaFiles if file[:10] == "evolution_"]) + 1
        else:
            currEvolution = 1
        name = "evolution_" + str(currEvolution)
        evolutionPath = os.path.join(self.dnaPath, name)
        os.mkdir(evolutionPath)
        text = name.upper() + settings.getInfo() + "\n    Fitness: " + snakes.Snake.fitness.__doc__[9:-9]

        # write settings for this training session
        with open(os.path.join(evolutionPath, "settings.txt"), "w") as f:
            f.write(text)

        # initialize training parameters
        population, generations = settings.populationSize, settings.generations
        initialPopulation = [snakes.SmartSnake(layers=settings.networkArchitecture, **settings.snakeParams) for _ in range(population)]
        fitness = snakes.Snake.fitness
        task = game.playTrainingGame
        colorCross = snakes.Snake.mergeTraits
        snakeDNA = genetics.Genetics(initialPopulation, task, fitness, mergeTraits=None)

        # train each generation
        print("\nPOPULATION SIZE:", population, "\nGENERATIONS:", generations, "\n")
        for gen in range(1, generations + 1):
            timer = time()
            snakeDNA.evolve()
            elapsed = round(time() - timer, 2)
            snakeDNA.printGenStats(gen)
            print("\tTime elapsed:", elapsed, "secs")
            bestSnake = snakeDNA.generations[gen]["best"]["object"]
            
            if settings.displayTraining:
                game.playTrainingGame(bestSnake, render=True)  # best snake of gen plays game in GUI window

            # save data of generation to .../dna/evolution_x/generation_y/analytics.json
            generationPath = os.path.join(evolutionPath, "generation_" + str(gen))
            os.mkdir(generationPath)
            f = open(os.path.join(generationPath, "analytics.json"), "w")
            data = snakeDNA.getGenStats(gen)
            data["time"] = elapsed
            json.dump(data, f, indent=4)
            f.close()

            # saves neural net of best snake from generation to .../dna/evolution_x/generation_y/model.npz
            modelPath = os.path.join(generationPath, "model.npz")
            np.savez(
                modelPath,
                weights=np.array(bestSnake.behavior.weights, dtype=object),
                biases=np.array(bestSnake.behavior.biases, dtype=object),
                color=np.array(bestSnake.color)
            )
            print()

    def _watchReplay(self) -> None:
        """Gets data from last game and replays it in GUI window"""
        if self.environment is None:
            print("\nNo game to re-watch!")
        else:
            data = self.environment.getData()
            self._replay(data["moves"], data["origin"], data["food"], data["mapSize"], data["color"])

    def _watchSaved(self) -> None:
        """Allows user to select saevd game from .../replays and replays it in GUI window"""
        replayFiles = os.listdir(self.replayPath)
        numReplays = len(replayFiles)
        if len(replayFiles) == 0:
            print("\nNo saved replays!")
        else:
            print()
            msg = "Select game to replay:"
            for i, gameSave in enumerate(replayFiles, start=1):
                msg += "\n\t" + str(i) + ") " + str(gameSave)
            backIndex = numReplays + 1
            msg += "\n\t" + str(backIndex) + ") Back"
            index = Driver.getValidInput(msg, dtype=int, valid=range(1, numReplays + 2)) - 1

            if index != backIndex-1:
                gameName = replayFiles[index]

                with open(self.replayPath + "/" + gameName) as f:
                    gameData = json.load(f)

                moves = [tuple(move) for move in gameData["moves"]]
                origin = tuple(gameData["origin"])
                food = [tuple(food) for food in gameData["food"]]
                mapSize = tuple(gameData["mapSize"])
                color = tuple(gameData["color"])
                self._replay(moves, origin, food, mapSize, color)

    def _replay(self, moves: list, origin: tuple, food: list, mapSize: tuple, color: tuple) -> None:
        """
        Constructs snake and environment from data and replays it in GUI window.

        Parameters
        ----------
        moves: list
            Queue of directions snake chose to make at each time step
        origin: tuple
            Initial starting position of snake
        food: list
            Queue of food spawning positions
        mapSize: tuple
            (x, y) size of game map snake played on
        color: tuple
            Color of snake
        """
        snake = snakes.Ghost(moves, **settings.snakeParams, color=color)
        self.environment = environments.Environment(snake, mapSize, origin=origin, food=food)
        game.playGame(self.environment)
        print()
        self._checkSave()

    @staticmethod
    def _exit() -> None:
        """Ensures GUI windows are properly closed."""
        print("\nExiting!")
        sys.exit()

    @staticmethod
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

    @staticmethod
    def hasExt(name: str, ext: str) -> bool:
        """
        Checks to see if name (file name) ends in proper extension.

        Parameters
        ----------
        name: str
            Name to check
        ext: str
            Target extension

        Returns
        -------
        bool: true if name ends in ext else false
        """
        length = len(ext)
        return len(name) > length and name[-1 * length:] == ext
