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

from core import settings, games
from core.ui import ui
from core.game import environment, snakes
from core.genetics import training, plotting

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
	
	Public Methods
	--------------
	run() -> None
		Allows user to select mode, runs mode
	"""
	def __init__(self) -> None:
		"""Gets paths to necessary folders."""
		self.prevGameEnvironment = None
		
		self.paths = {}
		self.paths["current"] = os.getcwd()
		self.paths["data"] = os.path.join(self.paths["current"], "data")
		self.paths["replays"] = os.path.join(self.paths["data"], "replays")
		self.paths["dna"] = os.path.join(self.paths["data"], "dna")
		self.paths["trained"] = os.path.join(self.paths["data"], "trained")
		self.paths["neural_net"] = os.path.join(self.paths["trained"], "neural_net")
		self.paths["multi"] = os.path.join(self.paths["trained"], "multi")
		self.paths["hierarchical"] = os.path.join(self.paths["trained"], "hierarchical")
		
		print()
		print("        +" + "="*8 + "+")
		print("        |SNAKE AI|")
		print("        +" + "="*8 + "+")
		print()
		
		print("Initialized with the following settings:")
		print(settings.getInfo(), "\n")

	def run(self) -> None:
		"""Allows user to select mode, runs mode."""
		modes = [
			("Play Classic", self._playClassic),
			("Play AI", self._playAI),
			("Replay Last Game", self._watchReplay),
			("Watch Saved", self._watchSaved),
			("Train AI", self._trainAI),
			("Exit", lambda: sys.exit())
		]
		
		ui.runModes(modes)
			
			
	def _playClassic(self) -> None:
		"""Opens GUI window and lets user play Snake with keyboard controls."""
		snake = snakes.Snake.Player()
		gameEnvironment = environment.Environment(snake, settings.mapSize)
		print("Get ready...")
		games.playPlayerGame(gameEnvironment)
		finalScore = snake.score
		print("Final score: " + str(finalScore))
		print()
		ui.checkSave(gameEnvironment, self._saveGame)
		self.prevGameEnvironment = gameEnvironment
		
	def _playAI(self) -> None:
		"""User selects saved model from .../dna/trained. Opens GUI window and AI plays game."""
		# check npz format with same size networks or something like that? Cant save certain things...??
		
		algoIndex, algoChoice = ui.getSelection("Neural Network", "Multi", "Hierarchical", "Cycle", "Pathfinding", "Floodfill", "Back", msg="Select AI algorithm:")
		
		if algoChoice != "Back" and (snake := self._makeSnake(algoIndex, algoChoice)) is not None:
			numGames = ui.getValidInput("How many games should be played? (select one game to render)", dtype=int, lower=1, end="\n")
			
			scores = []			
			timer = time()
			
			if algoChoice in {"Multi", "Hierarchical"}:
				algos = snake.behavior.algoUsage.keys()
				avgUsage = {algo: 0 for algo in algos}
			
			for i in range(numGames):
				gameEnvironment = environment.Environment(snake, settings.mapSize)
				#gameEnvironment = environment.Environment(snake, settings.mapSize, origin=(3, 0))  # for cycle to win or get close when odd
				games.playGame(gameEnvironment, render=(not (numGames-1)))
				scores.append(snake.size)
				if algoChoice in {"Multi", "Hierarchical"}:
					for algo in algos:
						avgUsage[algo] += snake.behavior.algoUsage[algo]
				print("Game", str(i+1) + " snake size:", snake.size)
			elapsed = time() - timer
			if not numGames - 1:
				print()
				ui.checkSave(gameEnvironment, self._saveGame)
				self.prevGameEnvironment = gameEnvironment
			else:
				print("\nTime elapsed across", str(numGames) + " games:", round(elapsed, 5), "secs")
				print("Average time per game", round(elapsed/numGames, 5), "secs")
				print("Average snake score:", round(sum(scores)/numGames, 2))
				print("Scores std:", round(np.std(scores), 3))
				if algoChoice in {"Multi", "Hierarchical"}:
					avgUsage = {algo: round(avgUsage[algo]/numGames, 3) for algo in algos}
					print("Snake average algorithm use (n=" + str(numGames) + "):", avgUsage)
				print()

	def _watchReplay(self) -> None:
		"""Gets data from last game and replays it in GUI window"""
		if self.prevGameEnvironment is None:
			print("\nNo game to re-watch!")
		else:
			data = self.prevGameEnvironment.getData()
			self._replay(data["moves"][::-1], data["origin"], data["food"], data["mapSize"], data["color"])

	def _watchSaved(self) -> None:
		"""Allows user to select saevd game from .../replays and replays it in GUI window"""
		replayFiles = os.listdir(self.paths["replays"])
		numReplays = len(replayFiles)
		if len(replayFiles) == 0:
			print("\nNo saved replays!")
		else:
			msg = "Select game to replay:"
			for i, gameSave in enumerate(replayFiles, start=1):
				msg += "\n\t" + str(i) + ") " + str(gameSave)
			backIndex = numReplays + 1
			msg += "\n\t" + str(backIndex) + ") Back"
			index = ui.getValidInput(msg, dtype=int, valid=range(1, numReplays + 2)) - 1

			if index != backIndex-1:
				gameName = replayFiles[index]

				with open(self.paths["replays"] + "/" + gameName) as f:
					gameData = json.load(f)

				moves = [tuple(move) for move in gameData["moves"]]
				origin = tuple(gameData["origin"])
				food = [tuple(food) for food in gameData["food"]]
				mapSize = tuple(gameData["mapSize"])
				color = tuple(gameData["color"])
				self._replay(moves, origin, food, mapSize, color)


	def _trainAI(self) -> None:
		"""Trains AI snakes. Saves data and models in ../dna folder. Creates new folder for each training session."""
		# initialize training parameters
		population, generations = settings.populationSize, settings.generations
		
		# settings validation
		if population < 10:
			print("\nError: Population size must be at least 10. Change size in settings.py.")
			return
		
		algoIndex, algoChoice = ui.getSelection("Neural Network", "Multi", "Hierarchical", "Back", msg="Select training type:")
		if algoChoice != "Back" and (initialPopulation := self._makeInitialPopulation(algoIndex, algoChoice, population)) is not None:
			colorCross = None
			#colorCross = snakes.Snake.mergeTraits  # include color crossing
			print("Evaluating initial...")
			snakeDNA = training.Genetics(initialPopulation, games.playTrainingGame, mergeTraits=colorCross)
			evolution = plotting.EvolutionGraph()
			evolution.display()
			
			initialBest = snakeDNA.generation["best"]["object"]
			
			# make following into its own function!!!!!!!!!!
			print("Running initial trials...")
			trials = 100
			scores = []
			if algoChoice in {"Multi", "Hierarchical"}:
				algos = initialBest.behavior.algoUsage.keys()
				avgUsage = {algo: 0 for algo in algos}
			for _ in range(trials):
				games.playTrainingGame(initialBest, render=False)
				scores.append(initialBest.score)
				if algoChoice in {"Multi", "Hierarchical"}:
					for algo in algos:
						avgUsage[algo] += initialBest.behavior.algoUsage[algo]
			avgScore = round(sum(scores)/trials, 3)
			std = round(np.std(scores))
			#print("    Best snake scores, avg score (n=" + str(trials) + "): " + str(avgScore) + ",", std)
			if algoChoice in {"Multi", "Hierarchical"}:
				avgUsage = {algo: round(avgUsage[algo]/trials, 3) for algo in algos}
				#print("    Best snake average algorithm use (n=" + str(trials) + "):", avgUsage)
			
			genData = snakeDNA.getGenStats()
			genData.update({"time": 0})
			genData["sampleScore"] = avgScore
			genData["sampleStd"] = std
			
			if evolution.active():
				evolution.update(genData)
			
			trainingTimer = time()
			
			# initialize paths and files
			if len(dnaFiles := os.listdir(self.paths["dna"])) > 0:
				currEvolution = max([int(file[file.index("_")+1:]) for file in dnaFiles if file[:10] == "evolution_"]) + 1
			else:
				currEvolution = 1
	 
			evolutionPath = os.path.join(self.paths["dna"], "evolution_" + str(currEvolution))
			os.mkdir(evolutionPath)
			data = {
				"evolution": currEvolution,
				"settings": settings.getDictInfo(),
				"fitness": initialPopulation[0].fitness.__doc__[9:-9],
				"architecture": settings.networkArchitecture
			}

			# write settings for this training session
			ui.saveToJSON(data, os.path.join(evolutionPath, "settings.json"))
			
			# train each generation
			print("\nPOPULATION SIZE:", population, "\nGENERATIONS:", generations, "\n")
			for gen in range(1, generations + 1):
				timer = time()
				snakeDNA.evolve()
				
				elapsed = round(time() - timer, 2)
				elapsedTotal = round(time() - trainingTimer, 2)
				
				snakeDNA.printGenStats()
				print("    Current memory usage:", str(psutil.virtual_memory()[2]) + str("%"))
				print("    Generation took:", ui.formatTime(elapsed))
				print("    Total time elapsed:", ui.formatTime(elapsedTotal))
				print("    Time of day:", datetime.now().strftime("%H:%M:%S"))
				
				bestSnake = snakeDNA.generation["best"]["object"]  # debug delete
				
				# RECORD IDX OF LEADING SNAKE HERE OR FROM GENETIC
				trials = 100
				scores = []
				if algoChoice in {"Multi", "Hierarchical"}:
					algos = bestSnake.behavior.algoUsage.keys()
					avgUsage = {algo: 0 for algo in algos}
				for _ in range(trials):
					games.playTrainingGame(bestSnake, render=False)
					scores.append(bestSnake.score)
					if algoChoice in {"Multi", "Hierarchical"}:
						for algo in algos:
							avgUsage[algo] += bestSnake.behavior.algoUsage[algo]
				avgScore = round(sum(scores)/trials, 3)
				std = round(np.std(scores))
				print("    Best snake scores, avg score (n=" + str(trials) + "): " + str(avgScore) + ",", std)
				if algoChoice in {"Multi", "Hierarchical"}:
					avgUsage = {algo: round(avgUsage[algo]/trials, 3) for algo in algos}
					print("    Best snake average algorithm use (n=" + str(trials) + "):", avgUsage)

				# save data of generation to .../dna/evolution_x/generation_y/analytics.json
				generationPath = os.path.join(evolutionPath, "generation_" + str(gen))
				data = snakeDNA.getGenStats()
				data.update({"time": elapsed})
				
				os.mkdir(generationPath)
				ui.saveToJSON(data, os.path.join(generationPath, "analytics.json"))

				# saves neural net of best snake from generation to .../dna/evolution_x/generation_y/model.npz
				modelPath = os.path.join(generationPath, "model.npz")
				model = bestSnake.getBrain()
				model.update({"color": bestSnake.color})
				
				self._saveSnakeData(model, modelPath, algoChoice)
				
				data["sampleScore"] = avgScore
				data["sampleStd"] = std
				
				if evolution.active():
					evolution.update(data)
				
				if settings.displayTraining:
					games.playTrainingGame(bestSnake, render=False)  # best snake of gen plays game in GUI window
				print()
			evolution.cleanup()
			snakeDNA.cleanup()
			
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
		gameEnvironment = environment.Environment(snake, mapSize, origin=origin, food=food)
		games.playGame(gameEnvironment)
		ui.checkSave(gameEnvironment, self._saveGame)
		
	def _makeInitialPopulation(self, algoIndex, algoChoice, population):
		modelsPath = self.paths["neural_net"]
		trainedFiles = os.listdir(modelsPath)
		
		if len(trainedFiles) == 0 or (len(trainedFiles) < 3 and algoChoice == "Hierarchical"):
			print("No enough trained AI!\n")
			return
				
		algorithms = ["neural network", "multi", "hierarchical"]
		
		behaviorKwargs = {}
		behaviorArgs = []
		
		snakeKwargs = {
			"initialSize": settings.initialSnakeSize,
			"maxVision": settings.maxSnakeVision,
			"hungerFunc": settings.hungerFunc,
		}

		# MAKE CHOICE FOR USER TO GO BACK
		if algoChoice == "Multi":
			# print CHOICE, A FINE SELECTION!
			modelIndex, choice = ui.getSelection(*trainedFiles, msg="\nSelect AI to use for neural network:")
			modelFile = trainedFiles[modelIndex]
			modelPath = os.path.join(modelsPath, modelFile)
			data = self._loadSnakeData(modelPath, "Neural Network")
			snakeKwargs.update(data["snakeKwargs"])
			behaviorKwargs.update(data["behaviorKwargs"])
			
		elif algoChoice == "Hierarchical":
			networkData = []
			selected = set()
			for i in range(3):  # remove already selected items from list once chosen??
				modelIndex, modelChoice = ui.getSelection(*trainedFiles, "Back", msg="\nSelect AI not already chosen to use:", isValid = lambda idx: not (idx - 1) in selected)
				if modelChoice == "Back":
					return
				selected.add(modelIndex)
				modelFile = trainedFiles[modelIndex]
				modelPath = os.path.join(modelsPath, modelFile)
				data = self._loadSnakeData(modelPath, "Neural Network")
				rawBehaviorKwargs = data["behaviorKwargs"]
				networkData.append({
					"weights": rawBehaviorKwargs["weights"],
					"biases": rawBehaviorKwargs["biases"],
					"architecture": rawBehaviorKwargs["architecture"],
				})
			
			behaviorKwargs.update({
				"networkData": networkData,
				"shielded": settings.smartShield
			})		
			
		else:  # training is neural networks
			behaviorKwargs.update({"architecture": settings.networkArchitecture})
			
		return [snakes.Snake(algorithms[algoIndex], behaviorArgs=behaviorArgs, behaviorKwargs=behaviorKwargs, id=np.base_repr(i+1, 36), **snakeKwargs) for i in range(population)]	

	def _makeSnake(self, algoIndex, algoChoice):
		algorithms = ["neural network", "multi", "hierarchical", "cycle", "pathfinder", "floodfill"]
		behaviorKwargs = {}
		behaviorArgs = []
		
		# make behavior args
		snakeKwargs = {
			"initialSize": settings.initialSnakeSize,
			"maxVision": settings.maxSnakeVision,
			"hungerFunc": settings.hungerFunc,
		}
		
		if algoChoice in {"Neural Network", "Multi", "Hierarchical"}:
			modelsPath = {
				"Neural Network": self.paths["neural_net"],
				"Multi": self.paths["multi"],
				"Hierarchical": self.paths["hierarchical"]
			}[algoChoice]
			
			trainedFiles = os.listdir(modelsPath)
			
			if len(trainedFiles) == 0:
				print("No trained AI!\n")
				return  # go back a page
				
			modelIndex, modelChoice = ui.getSelection(*trainedFiles, "Back", msg="Select AI to use:")
			
			if modelChoice == "Back":
				return self._playAI()
				
			modelFile = trainedFiles[modelIndex]
			modelPath = os.path.join(modelsPath, modelFile)
			
			data = self._loadSnakeData(modelPath, algoChoice)
			snakeKwargs.update(data["snakeKwargs"])
			behaviorArgs = data["behaviorArgs"]
			behaviorKwargs.update(data["behaviorKwargs"])
				
		elif algoChoice == "Pathfinding":
			behaviorKwargs = {"floodfill": True}
		elif algoChoice == "Floodfill":
			pass
		elif algoChoice == "Cycle":
			snakeKwargs["hungerFunc"] = lambda size: 1000  # so snkae does starve... shortcuts??

		snake = snakes.Snake(algorithms[algoIndex], behaviorArgs=behaviorArgs, behaviorKwargs=behaviorKwargs, **snakeKwargs)
		
		return snake
			
	def _loadSnakeData(self, path, algoChoice):
		data = np.load(path, allow_pickle=True)
		if algoChoice == "Neural Network":
			snakeKwargs = {"color": tuple([int(value) for value in data["color"]])}
			behaviorArgs = []
			behaviorKwargs = {
				"weights": [np.asarray(layer, dtype=float) for layer in data["weights"]],
				"biases": [np.asarray(layer, dtype=float) for layer in data["biases"]],
				"architecture": data["architecture"],
				"shielded": settings.smartShield
			}
		elif algoChoice == "Multi":
			snakeKwargs = {"color": tuple([int(value) for value in data["color"]])}
			behaviorArgs = []
			behaviorKwargs = {
				"weights": [np.asarray(layer, dtype=float) for layer in data["networkWeights"]],
				"biases": [np.asarray(layer, dtype=float) for layer in data["networkBiases"]],
				"architecture": data["networkArchitecture"],
				"metaWeights": [np.asarray(layer, dtype=float) for layer in data["metaWeights"]],
				"metaBiases": [np.asarray(layer, dtype=float) for layer in data["metaBiases"]],
				"metaArchitecture": data["metaArchitecture"],
				"shielded": settings.smartShield
			}
		elif algoChoice == "Hierarchical":
			snakeKwargs = {"color": tuple([int(value) for value in data["color"]])}
			behaviorArgs = []
			
			network1 = {"weights": data["networkWeights1"], "biases": data["networkBiases1"], "architecture": data["networkArchitecture1"]}
			network2 = {"weights": data["networkWeights2"], "biases": data["networkBiases2"], "architecture": data["networkArchitecture2"]}
			network3 = {"weights": data["networkWeights3"], "biases": data["networkBiases3"], "architecture": data["networkArchitecture3"]}
			
			networkData = [network1, network2, network3]
			
			behaviorKwargs = {
				"networkData": networkData,
				"metaWeights": [np.asarray(layer, dtype=float) for layer in data["metaWeights"]],
				"metaBiases": [np.asarray(layer, dtype=float) for layer in data["metaBiases"]],
				"metaArchitecture": data["metaArchitecture"],
				"shielded": settings.smartShield
			}
		else:
			raise NotImplementedError("Unknown algo choice: " + algoChoice)
		
		return {"behaviorKwargs": behaviorKwargs, "behaviorArgs": behaviorArgs, "snakeKwargs": snakeKwargs} 
		
	def _saveSnakeData(self, model, path, algoChoice):
		if algoChoice == "Neural Network":
			weights = np.array(model["weights"], dtype=object)
			biases = np.array(model["biases"], dtype=object)
			architecture = np.array(model["architecture"], dtype=object)
			np.savez(
				path,
				weights = weights,
				biases = biases,
				architecture = architecture,
				color = np.array(model["color"])
			)
		elif algoChoice == "Multi":
			metaWeights = np.array(model["weights"], dtype=object)
			metaBiases = np.array(model["biases"], dtype=object)
			metaArchitecture = np.array(model["metaArchitecture"], dtype=object)
			networkWeights = np.array(model["networkWeights"], dtype=object)
			networkBiases = np.array(model["networkBiases"], dtype=object)
			networkArchitecture = np.array(model["networkArchitecture"], dtype=object)
			np.savez(
				path,
				metaWeights = metaWeights,
				metaBiases = metaBiases,
				metaArchitecture = metaArchitecture,
				networkWeights = networkWeights,
				networkBiases = networkBiases,
				networkArchitecture = networkArchitecture,
				color = np.array(model["color"])
			)
		elif algoChoice == "Hierarchical":
			metaWeights = np.array(model["weights"], dtype=object)
			metaBiases = np.array(model["biases"], dtype=object)
			metaArchitecture = np.array(model["metaArchitecture"], dtype=object)
			networkWeights = [np.array(network["weights"], dtype=object) for network in model["networks"]]
			networkBiases = [np.array(network["biases"], dtype=object) for network in model["networks"]]
			networkArchitectures = [np.array(network["architecture"], dtype=object) for network in model["networks"]]
			np.savez(
				path,
				metaWeights = metaWeights,
				metaBiases = metaBiases,
				metaArchitecture = metaArchitecture,
				networkWeights1 = networkWeights[0],
				networkWeights2 = networkWeights[1],
				networkWeights3 = networkWeights[2],
				networkBiases1 = networkBiases[0],
				networkBiases2 = networkBiases[1],
				networkBiases3 = networkBiases[2],
				networkArchitecture1 = networkArchitectures[0],
				networkArchitecture2 = networkArchitectures[1],
				networkArchitecture3 = networkArchitectures[2],
				color = np.array(model["color"])
			)
		else:
			raise NotImplementedError("Unknown snake type: " + algoChoice)

	def _saveGame(self, environment, name):
		ui.saveToJSON(environment.getData(), os.path.join(self.paths["replays"], name))
			
