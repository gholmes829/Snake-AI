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
from datetime import datetime
import psutil

from core import genetics, game, environments, snakes, settings, behaviors

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
	modelPath:
		path to model folder
	seedPath:
		path to model/seeds
	
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
		self.seedPath = os.path.join(self.modelPath, "seeds")
		print("	   +" + "="*8 + "+")
		print("	   |SNAKE AI|")
		print("	   +" + "="*8 + "+")
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
		snake = snakes.Snake.Player(**settings.basicSnakeParams)
		self.environment = environments.Environment(snake, settings.mapSize)
		print("\nGet ready...")
		game.playPlayerGame(self.environment)
		print()
		self._checkSave()
		
	def _playAI(self, games=1, algorithmIndex=0) -> None:
		"""User selects saved model from .../dna/trained. Opens GUI window and AI plays game."""
		# ask user how many games?
		if not algorithmIndex:
			msg = "\nSelect AI algorithm:\n\t1) Neural Network\n\t2) Hybrid\n\t3) Cycle\n\t4) Pathfinding\n\t5) Back"
			algorithmIndex = Driver.getValidInput(msg, dtype=int, valid=range(1, 6))
		elif algorithmIndex > 5:
			raise IndexError("Algorithm index exceeds bounds")
		snakeKwargs = {
			**settings.basicSnakeParams,
			"hungerFunc": settings.hungerFunc,
		}
		if algorithmIndex == 5:
			return
		elif algorithmIndex in {1, 2}:
			trainedFiles = os.listdir(self.modelPath)
			trainedFiles.remove("seeds")
			numTrained = len(trainedFiles)
			if numTrained == 0:
				print("No trained AI!\n")
				return self._playAI()
			else:
				msg = "\nSelect AI to use:"
				for i, model in enumerate(trainedFiles, start=1):
					msg += "\n\t" + str(i) + ") " + str(model)
				msg += "\n\t" + str(len(trainedFiles)+1) + ") Back"
				index = Driver.getValidInput(msg, dtype=int, valid=range(1, numTrained + 2)) - 1
				if index == 4:
					return self._playAI()
				modelFile = trainedFiles[index]

				data = np.load(os.path.join(self.modelPath, modelFile), allow_pickle=True)
				color = tuple([int(value) for value in data["color"]])
				snakeKwargs["color"] = color
				behaviorKwargs = {
					"ctrlWeights": [np.asarray(layer, dtype=float) for layer in data["weights"]],
					"ctrlBiases": [np.asarray(layer, dtype=float) for layer in data["biases"]],
					"ctrlLayers": settings.networkArchitecture,
					"shielded": settings.smartShield
				}
				if algorithmIndex == 1:
					snake = snakes.Snake("neural network", behaviorKwargs=behaviorKwargs, **snakeKwargs)
				else:
					snake = snakes.Snake("hybrid", behaviorKwargs=behaviorKwargs, **snakeKwargs)
		elif algorithmIndex == 3:
			snake = snakes.Snake("cycle", **snakeKwargs)
		
		elif algorithmIndex == 4:
			snake = snakes.Snake("floodPathfinder", **snakeKwargs) 

		scores = []
		print()
		for i in range(games):
			self.environment = environments.Environment(snake, settings.mapSize, origin=(3, 0))
			game.playGame(self.environment, render=False and not (games-1))
			scores.append(snake.size)
			print("Final snake size for game", str(i+1) + ":", snake.size)

		if not games-1:
			print()
			self._checkSave()
		else:
			print("\nAverage snake size across", games, "games:", round(sum(scores)/games, 2))
			print()

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
		if settings.populationSize < 10:
			print("\nError: Population size must be at least 10. Change size in settings.py.")
			return

		# initialize training parameters
		population, generations = settings.populationSize, settings.generations

		msg = "\nSelect training type:\n\t1) NN Controller\n\t2) Meta Controller\n\t3) Back"
		trainingType = Driver.getValidInput(msg, dtype=int, valid={1, 2, 3})
		
		if trainingType == 3:
			return
		elif trainingType == 2:
		
			# EXPIRIMENTAL
			#print("META")  # delete
			trainedFiles = os.listdir(self.modelPath)
			trainedFiles.remove("seeds")
			numTrained = len(trainedFiles)

			modelFile = trainedFiles[3]  # hard coded to be king snek, fix later

			data = np.load(os.path.join(self.modelPath, modelFile), allow_pickle=True)
			color = tuple([int(value) for value in data["color"]])

			behaviorKwargs = {
				"ctrlWeights": [np.asarray(layer, dtype=float) for layer in data["weights"]],
				"ctrlBiases": [np.asarray(layer, dtype=float) for layer in data["biases"]],
				"ctrlLayers": settings.networkArchitecture,
				"shielded": settings.smartShield
			}
				
			snakeKwargs = {
				**settings.basicSnakeParams,
				"hungerFunc": settings.hungerFunc,
				"color": color
			}
			
			initialPopulation = [snakes.Snake("hybrid", behaviorKwargs=behaviorKwargs, **snakeKwargs) for _ in range(population)]
			
		else:
			#print("NN")  # delete
			seedFiles = os.listdir(self.seedPath)
			numSeeds = len(seedFiles)
			#numSeeds = 0
			print()
			snakeParams = {
				**settings.basicSnakeParams,
				"hungerFunc": settings.hungerFunc
			}
			choice = Driver.getValidInput("Select population seed:\n\t1) Random\n\t2) Starter seeds\n\t3) Back", dtype=int, valid=range(1, 4))
			if choice == 2 and numSeeds > 0:
				initialPopulation = []
				for modelFile in seedFiles[:-1]:
					data = np.load(os.path.join(self.modelPath, modelFile), allow_pickle=True)
					behaviorKwargs = {
						"ctrlLayers": settings.networkArchitecture,
						"ctrlWeights": [np.asarray(layer, dtype=float) for layer in data["weights"]],
						"ctrlBiases": [np.asarray(layer, dtype=float) for layer in data["biases"]]
					}

					clone = snakes.Snake("neural network", behaviorKwargs=behaviorKwargs, **settings.snakeParams)
					initialPopulation += [clone for _ in range(int(population/numSeeds)-1)] + [clone]
				data = np.load(os.path.join(self.modelPath, seedFiles[-1]), allow_pickle=True)
				behaviorKwargs = {
					"ctrlLayers": settings.networkArchitecture,
					"ctrlWeights": [np.asarray(layer, dtype=float) for layer in data["weights"]],
					"ctrlBiases": [np.asarray(layer, dtype=float) for layer in data["biases"]]
				}
				clone = snakes.Snake("neural network", behaviorKwargs=behaviorKwargs, **settings.snakeParams)
				initialPopulation += [clone for _ in range(population-len(initialPopulation)-1)] + [clone] 
			elif choice == 2 and numSeeds == 0:
				print("\nNo starter seeds in .../trained/seeds/")
				return
			elif choice == 3:
				return
			else:
				initialPopulation = [snakes.Snake("neural network", behaviorKwargs={"ctrlLayers": settings.networkArchitecture}, **snakeParams) for _ in range(population)]
		
		task = game.playTrainingGame
		colorCross = snakes.Snake.mergeTraits
		snakeDNA = genetics.Genetics(initialPopulation, task, mergeTraits=None)
		trainingTimer = time()
		
		# initialize paths and files
		dnaFiles = os.listdir(self.dnaPath)
		if len(dnaFiles) > 0:
			currEvolution = max([int(file[file.index("_")+1:]) for file in dnaFiles if file[:10] == "evolution_"]) + 1
		else:
			currEvolution = 1
 
		evolutionPath = os.path.join(self.dnaPath, "evolution_" + str(currEvolution))
		os.mkdir(evolutionPath)
		data = {
			"evolution": currEvolution,
			"settings": settings.getDictInfo(),
			"fitness": initialPopulation[0].fitness.__doc__[9:-9],
			"architecture": settings.networkArchitecture
		}

		# write settings for this training session
		with open(os.path.join(evolutionPath, "settings.json"), "w") as f:
			json.dump(data, f, indent=4)
		
		
		# train each generation
		print("\nPOPULATION SIZE:", population, "\nGENERATIONS:", generations, "\n")
		for gen in range(1, generations + 1):
			timer = time()
			snakeDNA.evolve()
			
			elapsed = round(time() - timer, 2)
			elapsedTotal = round(time() - trainingTimer, 2)
			currentTime = datetime.now().strftime("%H:%M:%S")
			
			generationTime = str(int(elapsed//3600)) + " hrs " + str(int((elapsed//60)%60)) + " mins " + str(int(elapsed%60)) + " secs"
			totalTime = str(int(elapsedTotal//3600)) + " hrs " + str(int((elapsedTotal//60)%60)) + " mins " + str(int(elapsedTotal%60)) + " secs"
			
			snakeDNA.printGenStats()
			print("	Current memory usage:", str(psutil.virtual_memory()[2]) + str("%"))
			print("	Generation took:", generationTime)
			print("	Total time elapsed:", totalTime)
			print("	Time of day:", currentTime)
			bestSnake = snakeDNA.generation["best"]["object"]
			
			# EXPIRIMENTAL
			#if trainingType == 2:  # delete
				#print(bestSnake.behavior.algorithmCount)
				#print(max(bestSnake.behavior.algorithmCount, key=bestSnake.behavior.algorithmCount.get))
			
			if settings.displayTraining:
				game.playTrainingGame(bestSnake, render=False)  # best snake of gen plays game in GUI window

			# save data of generation to .../dna/evolution_x/generation_y/analytics.json
			generationPath = os.path.join(evolutionPath, "generation_" + str(gen))
			os.mkdir(generationPath)
			f = open(os.path.join(generationPath, "analytics.json"), "w")
			data = snakeDNA.getGenStats()
			data["time"] = elapsed
			json.dump(data, f, indent=4)
			f.close()

			# saves neural net of best snake from generation to .../dna/evolution_x/generation_y/model.npz
			modelPath = os.path.join(generationPath, "model.npz")
			model = bestSnake.getBrain()
			#print(len(model["weights"]))
			#for l in model["weights"]:
				#print(l.shape)
			weights = np.array(model["weights"], dtype=object)
			biases = np.array(model["biases"], dtype=object)
			#print(weights.shape)
			#print(biases.shape)
			np.savez(
				modelPath,
				weights=weights,
				biases=biases,
				color=np.array(bestSnake.color)
			)
			print()
		snakeDNA.cleanup()

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
		snake = snakes.Snake("ghost", behaviorArgs=[moves], **settings.basicSnakeParams, color=color)
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
