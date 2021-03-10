#!/usr/bin/env python3
"""
Quick script to render and save figures.

Figures:
--------
G
"""

import os

import numpy as np
import json
from matplotlib import pyplot as plt

__author__ = "Grant Holmes"
__email__ = "g.holmes429@gmail.com"


def generationGraphs():
    path = os.path.join(os.getcwd(), "data/evolution")
    generations = os.listdir(path)
    generations.remove("settings.txt")
    generations.sort(key=lambda fn: int(fn[fn.index("_")+1:]))

    processed = {
        "generations": [],
        "bestScore": [],
        "avgScore": [],
        "bestFitness": [],
        "avgFitness": [],
        "trainingTime": []
    }

    for i, gen in enumerate(generations, start=1):
        with open(path + "/" + str(gen) + "/analytics.json") as f:
            data = json.load(f)
            processed["generations"].append(i)
            processed["bestScore"].append(data["highScore"])
            processed["avgScore"].append(np.mean(data["scores"]))
            processed["bestFitness"].append(data["fitnesses"][0])
            processed["avgFitness"].append(np.mean(data["fitnesses"]))
            processed["trainingTime"].append(data["time"])

    for key in ["Score", "Fitness"]:
        fig, ax = plt.subplots()
        ax.plot(processed["generations"], processed["best"+key], "o", c="red", label="Best")
        ax.plot(processed["generations"], processed["avg"+key], "o", c="cyan", label="Avgerage")
        ax.grid()
        ax.set_xlabel("Generation")
        ax.set_ylabel(key)
        ax.set_title(key+" vs Generation")
        ax.legend(loc="upper left")
        plt.savefig("figures/" + key.lower(), bbox_inches="tight")

    fig, ax = plt.subplots()
    ax.plot(processed["generations"], processed["trainingTime"], "o", c="red")
    ax.grid()
    ax.set_xlabel("Generation")
    ax.set_ylabel("Training Time (sec)")
    ax.set_title("Training Time vs Generation")
    plt.savefig("figures/" + "training_time", bbox_inches="tight")


def hiddenGraphs():
    paths = [os.path.join(os.getcwd(), "data/hidden_"+str(i)) for i in range(3)]

    generations = [os.listdir(path) for path in paths]
    for files in generations:
        files.remove("settings.txt")
        files.sort(key=lambda fn: int(fn[fn.index("_") + 1:]))

    processed = {
        "generations": list(range(1, len(generations[0])+1)),
        "hidden0": [],
        "hidden1": [],
        "hidden2": []
    }

    for i, path in enumerate(paths):
        for gen in generations[i]:
            with open(path + "/" + str(gen) + "/analytics.json") as f:
                data = json.load(f)
                processed["hidden"+str(i)].append(data["highScore"])

    fig, ax = plt.subplots()
    ax.plot(processed["generations"], processed["hidden0"], "o", c="red", label="Single Layer")
    ax.plot(processed["generations"], processed["hidden1"], "o", c="green", label="1 Hidden Layer")
    ax.plot(processed["generations"], processed["hidden2"], "o", c="cyan", label="2 Hidden Layers")
    ax.grid()
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Score")
    ax.set_title("Best Score vs Generation")
    ax.legend(loc="upper left")
    plt.savefig("figures/" + "layers", bbox_inches="tight")


def populationGraph():
    popPath = os.path.join(os.getcwd(), "data/populations")
    pops = os.listdir(popPath)
    rawData = {}
    for directory in pops:
        rawData[directory] = {
            "bestScore": 0,
            "avgTrainingTime": 0,
            "population": int(directory[directory.index("_")+1:]),
            "avgBestScore": 0,
        }

    paths = [os.path.join(popPath, directory) for directory in rawData.keys()]
    generations = [os.listdir(path) for path in paths]

    for files in generations:
        files.remove("settings.txt")
        files.sort(key=lambda fn: int(fn[fn.index("_") + 1:]))

    for i, path in enumerate(paths):
        highestScore = 0
        genBestScore = 0
        trainingTimes = []
        for gen in generations[i]:  # for each generation in pop_x dir
            with open(path + "/" + str(gen) + "/analytics.json") as f:
                data = json.load(f)
                score = data["highScore"]
                genBestScore += score
                if score > highestScore:
                    highestScore = score
                trainingTimes.append(data["time"])
        rawData[pops[i]]["avgBestScore"] = genBestScore/len(generations[i])
        rawData[pops[i]]["bestScore"] = highestScore
        rawData[pops[i]]["avgTrainingTime"] = np.mean(trainingTimes)

    populations = []
    training = []
    bestScores = []
    avgScores = []
    for val in rawData.values():
        populations.append(val["population"])
        training.append(val["avgTrainingTime"])
        bestScores.append(val["bestScore"])
        avgScores.append(val["avgBestScore"])

    order = np.argsort(populations)
    populations = np.array(populations)[order]
    training = np.array(training)[order]
    bestScores = np.array(bestScores)[order]
    avgScores = np.array(avgScores)[order]

    fig, axes = plt.subplots(2, sharex=True)
    axes[0].plot(populations, bestScores, "o", c="red", label="Best")
    axes[0].plot(populations, avgScores, "o", c="cyan", label="Average")
    axes[0].legend(loc="upper left")
    axes[0].set_title("Best Score")
    axes[0].set_ylabel("Score")
    axes[0].grid()
    axes[1].plot(populations, training, "o", c="green")
    axes[1].set_title("Avg Training Time")
    axes[1].set_ylabel("Time (sec)")
    axes[1].grid()
    axes[1].set_xlabel("Population Size")

    plt.savefig("figures/" + "population", bbox_inches="tight")

def confidenceGraph():
    # play game, gather size and behavior data
    pass


def main():
    """Main func."""
    plt.style.use(["dark_background"])
    plt.rc("grid", linestyle="dashed", color="white", alpha=0.25)
    generationGraphs()
    #hiddenGraphs()
    #populationGraph()
    #confidenceGraph()

    plt.show()


if __name__ == "__main__":
    main()
