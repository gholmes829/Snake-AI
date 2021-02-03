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
from joblib import Parallel, delayed
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
    """
    def __init__(self,
                 initialPopulation: list,
                 task: callable,
                 fitness: callable,
                 mergeTraits: callable = None,
                 crossoverRate: float = 0.3,
                 mutationRate: float = 0.7,
                 trials: int = 4
                 ) -> None:
        """
        Initializes.

        Parameters
        ----------
        initialPopulation: list
            Gen 0 population
        task: callable
            Task members will learn to do
        fitness: callable
            Function that can evaluate MemberType that returns fitness score
        mergeTraits: callable, optional
            Function called on parents during cross over to merge non essential traits like colors or names
        crossoverRate: float, default=0.3
            Used to determine many children will be created every generation
        mutationRate: float, default=0.7
            Used to determine how many mutants will be created every generation
        trials: int, default=4
            Number of performances will be averaged to determine member fitness
        """
        self.population = initialPopulation
        self.task = task
        self.fitness = fitness
        self.mergeTraits = mergeTraits

        self.size = len(initialPopulation)
        self.crossovers = int(crossoverRate * self.size)
        self.mutations = int(mutationRate * self.size)

        self.trials = trials
        self.gen = 0

        self.best, self.bestFitness = None, 0
        self.generations = {
            0: {
                "best": {"object": None, "fitness": 0, "score": 0},
                "population": [{"object": member, "fitness": 0, "score": 0} for member in self.population]
            }
        }

    def evolve(self) -> None:
        """Evolves population."""
        #pr = Profile()
        #pr.enable()
        self.gen += 1
        self.generations[self.gen] = {}

        parents = self._selectParents(self.population)
        population = self.population + \
                    self._makeChildren(parents) + \
                    self._makeMutants(self.population) + \
					self._makeSuperMutants(self.population)

        self.generations[self.gen]["population"] = self._evaluate(population)
        self.generations[self.gen]["population"].sort(key=lambda member: member["fitness"], reverse=True)
        self.generations[self.gen]["best"] = self.generations[self.gen]["population"][0]

        if self.generations[self.gen]["best"]["fitness"] > self.bestFitness:
            self.best = self.generations[self.gen]["best"]["object"]
            self.bestFitness = self.generations[self.gen]["best"]["fitness"]

        self.population = self._epigenetics([m["object"] for m in self.generations[self.gen]["population"]][:self.size])
        #pr.disable()
        #stats = pstats.Stats(pr).sort_stats("cumulative")
        #stats.print_stats(30)

    def getGenStats(self, gen: int) -> dict:
        """
        Gets score and fitness stats from a given generation.

        Parameters
        ----------
        gen: int
            Generation to get stats from

        Returns
        -------
        dict: stats in form of {"highScore": highest score from generation, "fitnesses": list of fitnesses}
        """
        scores = [float(snake["score"]) for snake in self.generations[gen]["population"]]
        scores.sort(reverse=True)
        highScore = float(max(scores))
        fitnesses = [float(snake["fitness"]) for snake in self.generations[gen]["population"]]
        return {"highScore": highScore, "scores": scores, "fitnesses": fitnesses}

    def printGenStats(self, gen: int) -> None:
        """
        Prints analytics of given generation to terminal.

        Parameters
        ----------
        gen: int
            Generation to print stats from
        """
        maxScore = max([snake["score"] for snake in self.generations[gen]["population"]])
        fitnesses = [snake["fitness"] for snake in self.generations[gen]["population"]]
        top10Avg = np.mean(fitnesses[:int(0.1 * self.size)])
        top25Avg = np.mean(fitnesses[:int(0.25 * self.size)])
        avg = np.mean(fitnesses)
        best = np.max(fitnesses)
        print("RESULTS FOR GEN:", gen)
        print("    Highest score:", maxScore)
        print("    Best fitness:", round(best, 2))
        print("    Average top 10% fitness:", round(top10Avg, 2))
        print("    Average top 25% fitness:", round(top25Avg, 2))
        print("    Average fitness:", round(avg, 2))

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
		
        if parallelize:
            try:
                trials = [Parallel(n_jobs=settings.cores)(delayed(self.task)(member) for member in population) for _ in range(self.trials)]  # parallelized
            except KeyboardInterrupt:
                print("Recieved keyboard interrupt signal. Exiting!")
                sys.exit()
            except Exception as e:
                print("EXCEPTION:", e)
                print()
                currentTime = datetime.now()
                strTime = str(int(currentTime.hour%13 + int(currentTime.hour > 12))) + ":" + currentTime.strftime("%M:%S") + " " + str({0: "AM", 1: "PM"}[currentTime.hour>=12])
                print("WARNING: An exception during parallelized evaluation has occured at " + strTime + ". Attempting to restart evaluation without parallelization.\n")
                return self._evaluate(population, parallelize=False)
        else:
            trials = [[self.task(member) for member in population] for _ in range(self.trials)]  # non parallelized
		
        for i in range(len(population)):
            fitness = np.mean([self.fitness(trials[j][i]) for j in range(self.trials)])
            score = np.max([trials[j][i].score for j in range(self.trials)])
            members.append({"object": population[i], "fitness": fitness, "score": score})

        return members

    def _selectParents(self, population: list) -> list:
        """
        Provides list of parent candidates from population.

        Parameters
        ----------
        population: list
            Population to select parents from

        Returns
        -------
        list: list of selected parents
        """
        return [self._selectParent(population) for _ in range(self.crossovers)]

    def _selectParent(self, population: list) -> object:
        """
        Selects winner of 3 member tourney as parent.

        Parameters
        ----------
        population: list
            Population to select parent from

        Returns
        -------
        MemberType: selected parent
        """
        members = []
        while len(members) < 3:
            member = population[randint(0, self.size - 1)]
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
        for member in members:
            self.task(member)

        return members[max(range(len(members)), key = lambda i: self.fitness(members[i]))]

    def _makeChildren(self, population: list) -> list:
        """
        Make children from a given population.

        Parameters
        ----------
        population: list
            Population to make children from

        Returns
        -------
        list: list of created children
        """
        return [self._makeChild(population) for _ in range(self.crossovers)]

    def _makeChild(self, population: list) -> object:
        """
        Make child by crossing over two random members from better end of population.

        Parameters
        ----------
        population: list
            Population to select parents from.

        Returns
        -------
        MemberType: child
        """
        parents = []
        while len(parents) < 2:
            parent = population[randint(0, self.crossovers - 1)]
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

        self.task(child1), self.task(child2)
        if self.mergeTraits is not None:
            self.mergeTraits(child1, parent1, parent2)
            self.mergeTraits(child2, parent1, parent2)
        return {True: child1, False: child2}[self.fitness(child1) > self.fitness(child2)]

    def _makeMutants(self, population: list) -> list:
        """
        Make mutants from given population.

        Parameters
        ----------
        population: list
            Population to make mutants from

        Returns
        -------
        list: list of created mutants
        """
        return [self._mutate(population[randint(0, self.size - 1)]) for _ in range(self.mutations)]
        
    def _makeSuperMutants(self, population: list) -> list:
        """
        Make super mutants (mutants based off top performing members) from given population.

        Parameters
        ----------
        population: list
            Population to make mutants from

        Returns
        -------
        list: list of created super mutants
        """
        superMutants = []
		
        for _ in range(3):
            candidate, sponsor = population[randint(0, 5)], population[randint(0, 10)]
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
