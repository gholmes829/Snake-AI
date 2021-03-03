"""
Uses multi-core parallel processing to train neural networks through natural selection.

Classes
-------
Genetics

References: The psuedo-code of Genetics.evolve and Genetics.evaluate is loosely based on that 
			of https://github.com/valentinmace/snake/blob/master/genetic_algorithm.py.
"""

import sys

from copy import deepcopy
from random import choice, randint
import numpy as np
from multiprocessing import Pool
from datetime import datetime

#from cProfile import Profile
#import pstats

from core import settings

__author__ = "Grant Holmes"
__email__ = "g.holmes429@gmail.com"


class Genetics:
	"""
	Trains neural networks to perform given task.

		Algorithm has many similarities to biological concepts such as evolution, natural selection, genetic mutation,
	crossover, and epigenetics. Population performs a given task and then those that perform best are able to pass down
	the traits that controlled their aptitude for the task to the next generation. Crossover allows multiple members
	of the population to combine their traits to pass down, hopefully creating a balanced, well performing member of the
	next generation. New members of the population are subject to genetic mutations that diversify the population set.
	Not all mutations are positive, and the mutated individuals who are worse at the task die off. However, the
	continuation of this process eventually yields a diverse solution set composed of the individuals whose traits
	survived.

		While this algorithm can only provide a non-deterministic approximate solution to a given problem or task,
	it can solve many difficult and complex problems without explicit solutions given enough time and and a suitable
	fitness function.

	Attributes
	----------
	population: list
		Most recent population set
	task: callable
		Task members will learn to do
	fitness: callable
		Function that can evaluate MemberType that returns fitness score
	mergeTraits: callable
		Function called on parents during cross over to merge non essential traits like colors or names
	size: int
		Population size
	crossovers: int
		How many children will be created every generation
	mutations: int
		How many mutants will be created every generation
	trials: int
		Number of performances will be averaged to determine member fitness
	gen: int
		Current generation
	best: MemberType
		Best overall member, irregardless of generation
	bestFitness: float
		Fitness score of best member
	generations: dict
		Keys are generations number values take form:
		   gen: {
					"population": list of {"object": MemberType, "fitness": float, "score": float},
					"best": {"object": MemberType, "fitness": float, "score": float}
			}

	Public Methods
	--------------
	evolve() -> None:
		Evolves population.
	getGenStats(gen: int) -> dict:
		Gets score and fitness stats from a given generation.
	printGenStats(gen: int) -> None:
		Prints analytics of given generation to terminal.
	cleanup() -> None:
		Closes and joins processes.
	"""
	def __init__(self,
				 initialPopulation: list,
				 task: callable,
				 mergeTraits: callable = None,
				 crossoverRate: float = 0.3,
				 mutationRate: float = 0.7,
				 trials: int = 3
				 ) -> None:
		"""
		Initializes.

		Parameters
		----------
		initialPopulation: list
			Gen 0 population
		task: callable
			Task members will learn to do, should return dictionary with 'fitness' and 'score' keys
		mergeTraits: callable, optional
			Function called on parents during cross over to merge non essential traits like colors or names
		crossoverRate: float, default=0.3
			Used to determine many children will be created every generation
		mutationRate: float, default=0.7
			Used to determine how many mutants will be created every generation
		trials: int, default=4
			Number of performances will be averaged to determine member fitness
		"""
		self.pool = Pool(processes=settings.cores)
		
		self.population = initialPopulation
		self.task = task
		self.mergeTraits = mergeTraits

		self.size = len(initialPopulation)
		self.crossovers = int(crossoverRate * self.size)
		self.mutations = int(mutationRate * self.size)

		self.trials = trials
		self.gen = 0

		self.best, self.bestFitness = None, 0
		
		self.parents = []
		self.generation = {
			"best": {"object": None, "fitness": 0, "score": 0},
			"population": [{"object": member, "fitness": 0, "score": 0} for member in self.population]
		}

	def evolve(self) -> None:
		"""Evolves population."""
		#pr = Profile()
		#pr.enable()
		self.gen += 1
		population = self._makePopulation(self.population)
		self.generation["population"] = self._evaluate(population)
		self.generation["population"].sort(key=lambda member: member["fitness"], reverse=True)
		self.generation["best"] = self.generation["population"][0]

		if self.generation["best"]["fitness"] > self.bestFitness:
			self.best = self.generation["best"]["object"]
			self.bestFitness = self.generation["best"]["fitness"]

		self.population = self._epigenetics([m["object"] for m in self.generation["population"]][:self.size])
		#pr.disable()
		#stats = pstats.Stats(pr).sort_stats("cumulative")
		#stats.print_stats(30)

	def getGenStats(self) -> dict:
		"""
		Gets score and fitness stats from a given generation.

		Returns
		-------
		dict: stats in form of {"highScore": highest score from generation, "fitnesses": list of fitnesses}
		"""
		scores = [float(snake["score"]) for snake in self.generation["population"]]
		scores.sort(reverse=True)
		highScore = float(max(scores))
		fitnesses = [float(snake["fitness"]) for snake in self.generation["population"]]
		return {"highScore": highScore, "scores": scores, "fitnesses": fitnesses}

	def printGenStats(self) -> None:
		"""
		Prints analytics of given generation to terminal.

		"""
		scores = [snake["score"] for snake in self.generation["population"]]
		maxScore = max(scores)
		avgScore = np.mean(scores)
		
		fitnesses = [snake["fitness"] for snake in self.generation["population"]]
		top10AvgFitness = np.mean(fitnesses[:int(0.1 * self.size)])
		top25AvgFitness = np.mean(fitnesses[:int(0.25 * self.size)])
		avgFitness = np.mean(fitnesses)
		bestFitness = np.max(fitnesses)
		
		print("RESULTS FOR GEN:", self.gen)
		print("    Highest score:", maxScore)
		print("    Average score:", round(avgScore, 5))
		print("    Average fitness:", round(avgFitness, 5))
		print("    Best fitness:", round(bestFitness, 5))
		print("    Top 10% fitness:", round(top10AvgFitness, 5))
		print("    Top 25% fitness:", round(top25AvgFitness, 5))


	def cleanup(self) -> None:
		"""Cleans up multiprocessing pool"""
		self.pool.close()
		self.pool.join()
		
	def _evaluate(self, population: list, parallelize: bool = True) -> list:
		"""
		Uses multi-core parallel processing to evaluate performance of each member in population.

		Parameters
		----------
		population: list
			Population to evaluate members from
		parallelize: bool, default=True
			Whether to parallelize operation

		Returns
		-------
		list: list of members paired with their stats in form of {"object": MemberType, "fitness": float, "score": float}
		"""
		members = []
		
		if parallelize and settings.cores > 1:
			try:
				trials = [self.pool.map(self.task, population) for _ in range(self.trials)]  # parallelized
			except KeyboardInterrupt:
				print("Recieved keyboard interrupt signal. Exiting!")
				sys.exit()
			except Exception as e:  # failures with process serialization and synchronization
				print("EXCEPTION:", e)
				print()
				currentTime = datetime.now().strftime("%H:%M:%S")
				print("WARNING: An exception during parallelized evaluation has occured at " + currentTime + ". Attempting to restart evaluation without parallelization.\n")
				return self._evaluate(population, parallelize=False)
		else:
			trials = [[self.task(member) for member in population] for _ in range(self.trials)]  # non parallelized
		
		for i in range(len(population)):
			fitness = np.mean([trials[j][i]["fitness"] for j in range(self.trials)])
			score = np.max([trials[j][i]["score"] for j in range(self.trials)])
			members.append({"object": population[i], "fitness": fitness, "score": score})

		return members

	def _makePopulation(self, population: list, parallelize: bool = False) -> list:
		"""
		Uses multi-core parallel processing to generate population for generation.

		Parameters
		----------
		population: list
			Initial population to generate members from
		parallelize: bool, default=True
			Whether to parallelize operation

		Returns
		-------
		list: list of population members
		"""
		if parallelize and settings.cores > 1:
			try:
				self.parents = self.pool.starmap(self._selectParent, [() for _ in range(self.crossovers)])
				children = self.pool.starmap(self._makeChild, [() for _ in range(self.crossovers)])
			except KeyboardInterrupt:
				print("Recieved keyboard interrupt signal. Exiting!")
				sys.exit()
			except Exception as e:  # failures with process serialization and synchronization
				print("EXCEPTION:", e)
				print()
				currentTime = datetime.now().strftime("%H:%M:%S")
				print("WARNING: An exception during parallelized evaluation has occured at " + currentTime + ". Attempting to restart evaluation without parallelization.\n")
				return self._makePopulation(population, parallelize=False)
		else:
			self.parents = self._selectParents()
			children = self._makeChildren()
			
		population = self.population + \
					children + \
					self._makeMutants()
					
		return population
		
	def _selectParents(self) -> list:
		"""
		Provides list of parent candidates from population.

		Returns
		-------
		list: list of selected parents
		"""
		return [self._selectParent() for _ in range(self.crossovers)]

	def _selectParent(self) -> object:
		"""
		Selects winner of 3 member tourney as parent.

		Returns
		-------
		MemberType: selected parent
		"""
		members = []
		while len(members) < 3:
			member = self.population[randint(0, self.size - 1)]
			if member not in members:
				members.append(member)
		return self._tourney(*members)

	def _tourney(self, *members: list) -> object:
		"""
		Has members compete in task, selects member who performed the best.

		Parameters
		----------
		members: list
			Members to compete in tourney

		Returns
		-------
		MemberType: winner of tourney
		"""
		return members[np.argmax([self.task(member)["fitness"] for member in members])]

	def _makeChildren(self) -> list:
		"""
		Make children from a given population.

		Returns
		-------
		list: list of created children
		"""
		return [self._makeChild() for _ in range(self.crossovers)]

	def _makeChild(self) -> object:
		"""
		Make child by crossing over two random members from better end of population.
		
		Returns
		-------
		MemberType: child
		"""
		parents = []
		while len(parents) < 2:
			parent = self.parents[randint(0, self.crossovers - 1)]
			if parent not in parents:
				parents.append(parent)
		return self._crossover(parents[0], parents[1])

	def _crossover(self, parent1: object, parent2: object) -> object:
		"""
		Cross over traits from parents for new member.

		Decides to cross over parents' weights or biases. Selects a random neuron to crossover.

		Parameters
		----------
		parent1: MemberType
			Member of crossover operation
		parent2: MemberType
			Member of crossover operation
		Returns
		-------
		MemberType: member created from crossover
		"""
		child1, child2 = deepcopy(parent1), deepcopy(parent2)
		brain1, brain2 = child1.getBrain(), child2.getBrain()
		target = choice(("weights", "biases"))

		randLayer = randint(0, len(brain1[target]) - 1)
		randTargetElement = randint(0, len(brain1[target][randLayer]) - 1)
		temp = deepcopy(brain1)
		brain1[target][randLayer][randTargetElement] = brain2[target][randLayer][randTargetElement]
		brain2[target][randLayer][randTargetElement] = temp[target][randLayer][randTargetElement]

		fitness1, fitness2 = self.task(child1)["fitness"], self.task(child2)["fitness"]
		if self.mergeTraits is not None:
			self.mergeTraits(child1, parent1, parent2)
			self.mergeTraits(child2, parent1, parent2)
		return {True: child1, False: child2}[fitness1 > fitness2]

	def _makeMutants(self) -> list:
		"""
		Make mutants from given population.

		Returns
		-------
		list: list of created mutants
		"""
		weaklyMutated = [self._mutate(self.population[randint(0, self.size - 1)]) for _ in range(int(0.9 * self.mutations))]
		stronglyMutated = []
		for _ in range(int(0.1 * self.mutations)):
			mutant = deepcopy(self.population[randint(0, self.size - 1)])
			for _ in range(randint(2, 4)):
				mutant = self._mutate(mutant)
			stronglyMutated.append(mutant)
		return weaklyMutated + stronglyMutated
		
	def _makeSuperMutants(self) -> list:
		"""
		Make super mutants (mutants based off top performing members) from given population.

		Returns
		-------
		list: list of created super mutants
		"""
		superMutants = []
		pool = self.population[:10]
		
		for i in range(3):
			candidate, sponsor = pool[randint(0, 4-i)], pool[randint(0, 9-i)]
			pool.remove(candidate)
			superMutants.append(self._crossover(self._mutate(candidate), sponsor))
		
		return superMutants

	def _mutate(self, member: object) -> object:
		"""
		Provides mutated clone of member.

		Mutates by assigning random value to a single weight or bias.

		Parameters
		----------
		member: MemberType
			Member to create mutated clone from

		Returns
		-------
		MemberType: mutated clone of member
		"""
		mutant = deepcopy(member)
		brain = mutant.getBrain()
		target = choice(("weights", "biases"))

		if target == "weights":
			randLayer = randint(0, len(brain["weights"]) - 1)
			randNeuron = randint(0, len(brain["weights"][randLayer]) - 1)
			randWeight = randint(0, len(brain["weights"][randLayer][randNeuron]) - 1)
			brain["weights"][randLayer][randNeuron][randWeight] = np.random.randn()
		else:
			randLayer = randint(0, len(brain["biases"]) - 1)
			randBias = randint(0, len(brain["biases"][randLayer]) - 1)
			brain["biases"][randLayer][randBias] = np.random.randn()

		return mutant

	def _epigenetics(self, population: list) -> list:
		"""
		Mimics epigenetics by randomly directly mutating population at fixed rate.

		Parameters
		----------
		population: list
			Population undergoing epigenetics

		Returns
		-------
		list: population after undergoing epigenetic mutations
		"""
		for _ in range(int(0.1 * self.size)):
			index = randint(10, self.size - 1)
			member = population[index]
			population[index] = self._mutate(member)

		return population
		
	def __getstate__(self) -> dict:
		internalState = self.__dict__.copy()
		del internalState["pool"]
		return internalState
		
	def __setstate__(self, state: dict) -> None:
		self.__dict__.update(state)
