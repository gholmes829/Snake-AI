"""
Sensitivity analysis, under development...
"""

import os
import numpy as np
from time import time

from core.game import environment, snakes, behaviors
from core import settings, games
from core.ui import ui

paths = {}
paths["current"] = os.getcwd()
paths["data"] = os.path.join(paths["current"], "data")
paths["replays"] = os.path.join(paths["data"], "replays")
paths["dna"] = os.path.join(paths["data"], "dna")
paths["trained"] = os.path.join(paths["data"], "trained")
paths["neural_net"] = os.path.join(paths["trained"], "neural_net")
paths["multi"] = os.path.join(paths["trained"], "multi")
paths["hierarchical"] = os.path.join(paths["trained"], "hierarchical")

def main():
    algoIndex, algoChoice = ui.getSelection("Neural Network", "Multi", "Hierarchical", "Cycle", "Pathfinding", "Floodfill", msg="Select AI algorithm:")
    
    if (snake := makeSnake(algoIndex, algoChoice)) is not None:
        #numGames = ui.getValidInput("How many games should be played?", dtype=int, lower=1, end="\n")
        S = []  # true states
        P = []  # perturbed states
        
        scores = []            
        timer = time()
        
        if algoChoice in {"Multi", "Hierarchical"}:
            algos = snake.behavior.algoUsage.keys()
            avgUsage = {algo: 0 for algo in algos}
        
        numGames = 1  # hard coded in for now
        for i in range(numGames):
            gameEnvironment = environment.Environment(snake, settings.mapSize, noise=settings.noise)

            while gameEnvironment.active():
                gameEnvironment.step()
                S.append(gameEnvironment.gameMap.copy())  # adding true state
            
            scores.append(snake.size)
            if algoChoice in {"Multi", "Hierarchical"}:
                for algo in algos:
                    avgUsage[algo] += snake.behavior.algoUsage[algo]
            print("Game", str(i+1) + " snake size:", snake.size)
        elapsed = time() - timer

        print("\nTime elapsed across", str(numGames) + " games:", round(elapsed, 5), "secs")
        print("Average time per game", round(elapsed/numGames, 5), "secs")
        print("Average snake score:", round(sum(scores)/numGames, 2))
        print("Scores std:", round(np.std(scores), 3))
        if algoChoice in {"Multi", "Hierarchical"}:
            avgUsage = {algo: round(avgUsage[algo]/numGames, 3) for algo in algos}
            print("Snake average algorithm use (n=" + str(numGames) + "):", avgUsage)
        print()
        
        P = [perturb(s) for s in S]  # independently perturb each true state
        
        radius = 5  # threshold for which to search for solutions
        solutions = solve(P, radius)  # get a list of solutions within radius
        
        if S in solutions:
            print("Converged!")
        elif len(solutions) == 0:
            print("Failed to find any solutions")
        else:
            print("Failed to converge to true trajectory")
        
        print()
    print("Done!")
    
def perturb(s, radius, perturbationType="default"):
    """
    
    """
    p = s  # actually perturb
    return p
        
def solve(P, radius):
    solutions = []
    # find solutions
    return P
    
def metric(s1, s2, metricType="default"):
    """
    Same length
    Orientation
    Euclidean distance
    Distance between each component, weight each component
    Flood fill open space?
    Test different metrics
    1, 1/2, 1/4, ...
    """
    return 0
        
def makeSnake(algoIndex, algoChoice):
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
            "Neural Network": paths["neural_net"],
            "Multi": paths["multi"],
            "Hierarchical": paths["hierarchical"]
        }[algoChoice]
        
        trainedFiles = os.listdir(modelsPath)
        trainedFiles.remove(".gitkeep")
        
        if len(trainedFiles) == 0:
            print("No trained AI for choice!\n")
            return  # go back a page
            
        modelIndex, modelChoice = ui.getSelection(*trainedFiles, msg="Select AI to use:")
            
        modelFile = trainedFiles[modelIndex]
        modelPath = os.path.join(modelsPath, modelFile)
        
        data = loadSnakeData(modelPath, algoChoice)
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
            
def loadSnakeData(path, algoChoice):
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

if __name__ == "__main__":
    main()