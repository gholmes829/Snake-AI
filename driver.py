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

from core import genetics, game, environments, snakes, settings, behaviors, util

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
		
		print("	   +" + "="*8 + "+")
		print("	   |SNAKE AI|")
		print("	   +" + "="*8 + "+")
		
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
			index, _ = util.getSelection("Play Classic", "Play AI", "Train AI", "Replay Last Game", "Watch Saved", "Exit", msg="Select mode:")
			modes[index]()
			print()
			
			
	def _playClassic(self) -> None:
		"""Opens GUI window and lets user play Snake with keyboard controls."""
		self.currGameEnvironment = environments.Environment(snakes.Snake.Player(), settings.mapSize)
		print()
		game.playPlayerGame(self.currGameEnvironment)
		print()
		self._checkSave()
		
	def _playAI(self, games=10) -> None:
		"""User selects saved model from .../dna/trained. Opens GUI window and AI plays game."""
		# ask user how many games?
		# change snake kwargs
		# check npz format with same size networks or something like that? Cant save certain things...??
		behaviorsKwargs = {}
		snakeKwargs = {
			"initialSize": settings.initialSnakeSize,
			"maxVision": settings.maxSnakeVision,
			"hungerFunc": settings.hungerFunc,
		}
		
		algorithms = ["neural network", "hybrid", "cycle", "floodPathfinder"]
		algoIndex, choice = util.getSelection("Neural Network", "Hybrid", "Cycle", "Pathfinding", "Back", msg="Select AI algorithm:")
		
		if choice == "Back":
			return
		elif choice == "Neural Network":
			trainedFiles = os.listdir(self.modelPath)
			trainedFiles.remove("seeds")
			if len(trainedFiles) == 0:
				print("No trained AI!\n")
				return self._playAI()  # go back a page in a sense
			modelIndex, modelChoice = util.getSelection(*trainedFiles, "Back", msg="Select AI to use:")
			if modelChoice == "Back":
				return self._playAI()
			modelFile = trainedFiles[modelIndex]
			modelPath = os.path.join(self.modelPath, modelFile)
			data = self._loadSnakeData(modelPath)
			snakeKwargs["color"] = tuple([int(value) for value in data["color"]])
			behaviorKwargs = {
				"ctrlWeights": data["weights"],
				"ctrlBiases": data["biases"],
				"ctrlLayers": data["architecture"],
				"shielded": settings.smartShield
			}
		elif choice == "Hybrid":
			raise NotImplementedError

		snake = snakes.Snake(algorithms[algoIndex], behaviorKwargs=behaviorKwargs, **snakeKwargs)
			
		scores = []
		print()
		for i in range(games):
			self.currGameEnvironment = environments.Environment(snake, settings.mapSize)
			#self.currGameEnvironment = environments.Environment(snake, settings.mapSize, origin=(3, 0))  for cycle to win or get close when odd
			game.playGame(self.currGameEnvironment, render=(not (games-1)))
			scores.append(snake.size)
			print("Final snake size for game", str(i+1) + ":", snake.size)

		if not games-1:
			print()
			self._checkSave()
		else:
			print("\nAverage snake size across", games, "games:", round(sum(scores)/games, 2))
			print()

	def _loadSnakeData(self, path):
		data = util.loadNPZ(path)
		return {
			"weights": [np.asarray(layer, dtype=float) for layer in data["weights"]],
			"biases": [np.asarray(layer, dtype=float) for layer in data["biases"]],
			"architecture": data["architecture"],
			"color": data["color"] 
		}
	def _checkSave(self) -> None:
		"""Checks to see if user wants to save last game to .../replays. If so, user inputs file name."""
		index, _ = util.getSelection("Yes", "No", msg="Save game as replay?")
		if not index:
			print()
			self._saveGame(util.getValidInput("File name?"))

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
		
		algoIndex, algoChoice = util.getSelection("NN Controller", "Meta Controller", "Back", msg="Select training type:")
		if algoChoice == "Back":
			return

		trainedFiles = os.listdir(self.modelPath)
		trainedFiles.remove("seeds")
		
		if len(trainedFiles) == 0:
			print("No trained AI!\n")
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
			bestSnake = snakeDNA.generation["best"]["object"]
			
			# RECORD IDX OF LEADING SNAKE HERE OR FROM GENETIC
			
			#EXPIRIMENTAL
			if algoChoice == "Meta Controller":  # delete
				game.playTrainingGame(bestSnake, render=False)
				avgAlgos = {}
				for algo in bestSnake.behavior.algorithmCount:
					avgAlgos[algo] = 0
					for snakeData in snakeDNA.generation["population"]:
						avgAlgos[algo] += snakeData["object"].behavior.algorithmCount[algo]
					avgAlgos[algo] = round(avgAlgos[algo] / len(snakeDNA.generation["population"]), 2)
				print("    Average algorithm use:", avgAlgos)
				print("    Best snake algorithm use:", bestSnake.behavior.algorithmCount, bestSnake.score) 
			
			if settings.displayTraining:
				game.playTrainingGame(bestSnake, render=False)  # best snake of gen plays game in GUI window

			# save data of generation to .../dna/evolution_x/generation_y/analytics.json
			generationPath = os.path.join(evolutionPath, "generation_" + str(gen))
			data = snakeDNA.getGenStats().update({"time": elapsed})
			self._logGenerationData(data, generationPath)

			# saves neural net of best snake from generation to .../dna/evolution_x/generation_y/model.npz
			modelPath = os.path.join(generationPath, "model.npz")
			model = bestSnake.getBrain()
			model.update({"color": bestSnake.color})
			
			self._saveSnakeData(model, modelPath)
			print()
		snakeDNA.cleanup()
		
	def _makeInitialPopulation(self, algoChoice, algoIndex, trainedFiles, population):
		algorithms = ["neural network", "hybrid"]
		
		behaviorKwargs = {}
		snakeKwargs = {
			"initialSize": settings.initialSnakeSize,
			"maxVision": settings.maxSnakeVision,
			"hungerFunc": settings.hungerFunc,
		}
		

		if algoChoice == "Meta Controller":

			# print CHOICE, A FINE SELECTION!
			modelIndex, choice = util.getSelection(*trainedFiles, "Back", msg="Select AI to use:")
			modelFile = trainedFiles[modelIndex]
			modelPath = os.path.join(self.modelPath, modelFile)
			data = self._loadSnakeData(modelPath)

			behaviorKwargs.update({
				"ctrlWeights": data["weights"],
				"ctrlBiases": data["biases"],
				"ctrlLayers": data["architecture"],
				"shielded": settings.smartShield
			})
			snakeKwargs["color"] = tuple([int(value) for value in data["color"]])
		else:  # training is neural networks
			behaviorKwargs.update({"ctrlLayers": settings.networkArchitecture})
			
		return [snakes.Snake(algorithms[algoIndex], behaviorKwargs=behaviorKwargs, **snakeKwargs) for _ in range(population)]	
		
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
			index = util.getValidInput(msg, dtype=int, valid=range(1, numReplays + 2)) - 1

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
