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
		self.currGameEnvironment = None
		self.replayPath = os.path.join(os.getcwd(), "replays")
		self.dnaPath = os.path.join(os.getcwd(), "dna")
		self.modelPath = os.path.join(os.getcwd(), "trained")
		self.seedPath = os.path.join(self.modelPath, "seeds")
		
		print()
		print("        +" + "="*8 + "+")
		#print("        |--------|")
		print("        |SNAKE AI|")
		#print("        |--------|")
		print("        +" + "="*8 + "+")
		
		print("\nInitialized with the following settings:")
		print(settings.getInfo(), "\n")

	def run(self) -> None:
		"""Allows user to select mode, runs mode."""
		modes = [
			self._playClassic,
			self._playAI,
			self._trainAI,
			self._watchReplay,
			self._watchSaved,
			Driver._exit
		]
		
		mode = None
		while mode != 6:
			index, _ = self.getSelection("Play Classic", "Play AI", "Train AI", "Replay Last Game", "Watch Saved", "Exit", msg="Select mode:")
			modes[index]()
			print()
			
			
	def _playClassic(self) -> None:
		"""Opens GUI window and lets user play Snake with keyboard controls."""
		self.currGameEnvironment = environments.Environment(snakes.Snake.Player(), settings.mapSize)
		print()
		game.playPlayerGame(self.currGameEnvironment)
		print()
		self._checkSave()
		
	def _playAI(self) -> None:
		"""User selects saved model from .../dna/trained. Opens GUI window and AI plays game."""
		# ask user how many games?
		# check npz format with same size networks or something like that? Cant save certain things...??
		behaviorKwargs = {}
		behaviorArgs = []
		
		# make behavior args
		snakeKwargs = {
			"initialSize": settings.initialSnakeSize,
			"maxVision": settings.maxSnakeVision,
			"hungerFunc": settings.hungerFunc,
		}
		
		algorithms = ["neural network", "multi", "hierarchical", "cycle", "pathfinder", "floodfill"]
		algoIndex, choice = self.getSelection("Neural Network", "Multi", "Hierarchical", "Cycle", "Pathfinding", "Floodfill", "Back", msg="\nSelect AI algorithm:")
		
		trainedFiles = os.listdir(self.modelPath)
		trainedFiles.remove("seeds")
		
		if choice == "Back":
			return
		elif choice in {"Neural Network", "Multi"}:
			if len(trainedFiles) == 0:
				print("No trained AI!\n")
				return self._playAI()  # go back a page in a sense
			modelIndex, modelChoice = self.getSelection(*trainedFiles, "Back", msg="\nSelect AI to use:")
			if modelChoice == "Back":
				return self._playAI()
			modelFile = trainedFiles[modelIndex]
			modelPath = os.path.join(self.modelPath, modelFile)
			data = self._loadSnakeData(modelPath)
			snakeKwargs["color"] = tuple([int(value) for value in data["color"]])
			behaviorKwargs.update({
				"weights": data["weights"],
				"biases": data["biases"],
				"architecture": data["architecture"],
				"shielded": settings.smartShield
			})
			if choice == "Multi":
				pass  # anything extra that multi needs
		elif choice == "Hierarchical":
			networkData = []
			selected = set()
			if len(trainedFiles) < 3:
				print("Not enough trained AI!\n")
				return self._playAI()  # go back a page in a sense
			for i in range(3):  # remove already selected items from list once chosen??
				modelIndex, modelChoice = self.getSelection(*trainedFiles, "Back", msg="\nSelect AI not already chosen to use:", isValid = lambda idx: not (idx-1) in selected)
				if modelChoice == "Back":
					return self._playAI()
				selected.add(modelIndex)
				modelFile = trainedFiles[modelIndex]
				modelPath = os.path.join(self.modelPath, modelFile)
				data = self._loadSnakeData(modelPath)
				networkData.append({
					"weights": data["weights"],
					"biases": data["biases"],
					"architecture": data["architecture"],
				})
			
			behaviorKwargs.update({
				"networkData": networkData,
				"shielded": settings.smartShield
			})			
				
		elif choice == "Pathfinding":
			behaviorKwargs = {"floodfill": True}
		elif choice == "Floodfill":
			pass
		elif choice == "Cycle":
			snakeKwargs["hungerFunc"] = lambda size: 1000  # so snkae does starve... shortcuts??

		snake = snakes.Snake(algorithms[algoIndex], behaviorArgs=behaviorArgs, behaviorKwargs=behaviorKwargs, **snakeKwargs)
			
		print()
		games = self.getValidInput("How many games should be played?", dtype=int, lower=1)
			
		scores = []
		print()
		timer = time()
		for i in range(games):
			self.currGameEnvironment = environments.Environment(snake, settings.mapSize)
			#self.currGameEnvironment = environments.Environment(snake, settings.mapSize, origin=(3, 0))  # for cycle to win or get close when odd
			game.playGame(self.currGameEnvironment, render=(not (games-1)))
			scores.append(snake.size)
			print("Game", str(i+1) + " snake size:", snake.size)
		elapsed = time() - timer
		if not games-1:
			print()
			self._checkSave()
			print()
		else:
			print("\nTime elapsed across", str(games) + " games:", round(elapsed, 5), "secs")
			print("Average snake score:", round(sum(scores)/games, 2))
			print("Scores std:", round(np.std(scores), 3))

	def _loadSnakeData(self, path):
		data = self.loadNPZ(path)
		return {
			"weights": [np.asarray(layer, dtype=float) for layer in data["weights"]],
			"biases": [np.asarray(layer, dtype=float) for layer in data["biases"]],
			"architecture": data["architecture"],
			"color": data["color"] 
		}
	def _checkSave(self) -> None:
		"""Checks to see if user wants to save last game to .../replays. If so, user inputs file name."""
		index, _ = self.getSelection("Yes", "No", msg="Save game as replay?")
		if not index:
			print()
			self._saveGame(self.getValidInput("File name?"))

	# DEF REFACTOR GENETIC SPLITTING UP
			
	def _saveGame(self, name: str) -> None:
		"""
		Saves game environment to .json file in .../replay folder.

		Parameters
		----------
		name: str
			File name for saved game file
		"""
		if not (len(name) > len(".json") and name[-1 * len(".json"):] == ".json"):
			name += ".json"
		data = self.currGameEnvironment.getData()
		with open(os.path.join(self.replayPath, name), "w") as f:
			json.dump(data, f, indent=4)

	def _trainAI(self) -> None:
		"""Trains AI snakes. Saves data and models in ../dna folder. Creates new folder for each training session."""
		# initialize training parameters
		population, generations = settings.populationSize, settings.generations
		
		# settings validation
		if population < 10:
			print("\nError: Population size must be at least 10. Change size in settings.py.")
			return
		
		algoIndex, algoChoice = self.getSelection("NN Controller", "Multi Controller", "Hierarchical Controller", "Back", msg="\nSelect training type:")
		if algoChoice == "Back":
			return

		trainedFiles = os.listdir(self.modelPath)
		trainedFiles.remove("seeds")
		
		if len(trainedFiles) == 0:
			print("No trained AI!\n")
			return self._trainAI()
			
		if len(trainedFiles) < 3 and algoChoiec == "Hierarchical Controller":
			print("Not enough trained AI!\n")
			return self._trainAI()
			
		initialPopulation = self._makeInitialPopulation(algoChoice, algoIndex, trainedFiles, population)
		task = game.playTrainingGame
		colorCross = None
		#colorCross = snakes.Snake.mergeTraits  # include color crossing
		snakeDNA = genetics.Genetics(initialPopulation, task, mergeTraits=colorCross)
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
			print("    Current memory usage:", str(psutil.virtual_memory()[2]) + str("%"))
			print("    Generation took:", generationTime)
			print("    Total time elapsed:", totalTime)
			print("    Time of day:", currentTime)
			bestSnake = snakeDNA.generation["best"]["object"]  # debug delete
			
			# RECORD IDX OF LEADING SNAKE HERE OR FROM GENETIC
			
			
			trials = 10
			scores = []
			if algoChoice in {"Multi Controller", "Hierarchical Controller"}:
				algos = bestSnake.behavior.algoUsage.keys()
				avgUsage = {algo: 0 for algo in algos}
			for _ in range(trials):
				game.playTrainingGame(bestSnake, render=False)
				scores.append(bestSnake.score)
				if algoChoice in {"Multi Controller", "Hierarchical Controller"}:
					for algo in algos:
						avgUsage[algo] += bestSnake.behavior.algoUsage[algo]
			avgScore = round(sum(scores)/trials, 3)
			print("    Best snake scores, avg score (n=" + str(trials) + "):", str(scores) + ",", avgScore)
			if algoChoice in {"Multi Controller", "Hierarchical Controller"}:
				avgUsage = {algo: round(avgUsage[algo]/trials, 3) for algo in algos}
				print("    Best snake average algorithm use (n=" + str(trials) + "):", avgUsage)
			
			
			if settings.displayTraining:
				game.playTrainingGame(bestSnake, render=False)  # best snake of gen plays game in GUI window

			# save data of generation to .../dna/evolution_x/generation_y/analytics.json
			generationPath = os.path.join(evolutionPath, "generation_" + str(gen))
			data = snakeDNA.getGenStats()
			data.update({"time": elapsed})
			self._logGenerationData(data, generationPath)

			# saves neural net of best snake from generation to .../dna/evolution_x/generation_y/model.npz
			modelPath = os.path.join(generationPath, "model.npz")
			model = bestSnake.getBrain()
			model.update({"color": bestSnake.color})
			
			self._saveSnakeData(model, modelPath)
			print()
		snakeDNA.cleanup()
		
	def _makeInitialPopulation(self, algoChoice, algoIndex, trainedFiles, population):
		algorithms = ["neural network", "multi", "hierarchical"]
		
		behaviorKwargs = {}
		behaviorArgs = []
		
		snakeKwargs = {
			"initialSize": settings.initialSnakeSize,
			"maxVision": settings.maxSnakeVision,
			"hungerFunc": settings.hungerFunc,
		}

		# MAKE CHOICE FOR USER TO GO BACK
		if algoChoice == "Multi Controller":
			# print CHOICE, A FINE SELECTION!
			modelIndex, choice = self.getSelection(*trainedFiles, msg="\nSelect AI to use:")
			modelFile = trainedFiles[modelIndex]
			modelPath = os.path.join(self.modelPath, modelFile)
			data = self._loadSnakeData(modelPath)

			behaviorKwargs.update({
				"weights": data["weights"],
				"biases": data["biases"],
				"architecture": data["architecture"],
				"shielded": settings.smartShield
			})
			snakeKwargs["color"] = tuple([int(value) for value in data["color"]])
			
		elif algoChoice == "Hierarchical Controller":
			networkData = []
			selected = set()
			for i in range(3):  # remove already selected items from list once chosen??
				modelIndex, modelChoice = self.getSelection(*trainedFiles, msg="\nSelect AI not already chosen to use:", isValid = lambda idx: not (idx - 1) in selected)
				selected.add(modelIndex)
				modelFile = trainedFiles[modelIndex]
				modelPath = os.path.join(self.modelPath, modelFile)
				data = self._loadSnakeData(modelPath)
				networkData.append({
					"weights": data["weights"],
					"biases": data["biases"],
					"architecture": data["architecture"],
				})
			
			behaviorKwargs.update({
				"networkData": networkData,
				"shielded": settings.smartShield
			})		
			
		else:  # training is neural networks
			behaviorKwargs.update({"architecture": settings.networkArchitecture})
			
		return [snakes.Snake(algorithms[algoIndex], behaviorArgs=behaviorArgs, behaviorKwargs=behaviorKwargs, id=np.base_repr(i+1, 36), **snakeKwargs) for i in range(population)]	
		
	def _saveSnakeData(self, model, path):
		weights = np.array(model["weights"], dtype=object)
		biases = np.array(model["biases"], dtype=object)  # do these need to be converted from 
		architecture = np.array(model["architecture"], dtype=object)
		np.savez(
			path,
			weights=weights,
			biases=biases,
			architecture=architecture,
			color=np.array(model["color"])
		)

	def _logGenerationData(self, data, path):
		os.mkdir(path)
		f = open(os.path.join(path, "analytics.json"), "w")
		json.dump(data, f, indent=4)
		f.close()
		
	def _watchReplay(self) -> None:
		"""Gets data from last game and replays it in GUI window"""
		if self.currGameEnvironment is None:
			print("\nNo game to re-watch!")
		else:
			data = self.currGameEnvironment.getData()
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
			index = self.getValidInput(msg, dtype=int, valid=range(1, numReplays + 2)) - 1

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
		snakeKwargs = {
			"initialSize": settings.initialSnakeSize,
			"maxVision": settings.maxSnakeVision,
			"hungerFunc": settings.hungerFunc,
		}
		
		snake = snakes.Snake("ghost", behaviorArgs=[moves], **snakeKwargs, color=color)
		self.currGameEnvironment = environments.Environment(snake, mapSize, origin=origin, food=food)
		game.playGame(self.currGameEnvironment)
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
	def getSelection(*args, msg: str = "Choose item:", **kwargs) -> tuple:
		for i, item in enumerate(args):
			msg += "\n    " + str(i + 1) + ") " + str(item)
			
		i = Driver.getValidInput(msg, dtype=int, lower=1, upper=len(args), **kwargs) - 1
		return i, args[i]
	
	@staticmethod	
	def loadNPZ(path):
		 return np.load(path, allow_pickle=True)
		
