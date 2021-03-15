"""

"""

import numpy as np
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt

# turn this into an abstracted async graph?

class EvolutionGraph:
	def __init__(self):
		self.q = Queue()
		self.p = Process(target=self._render, args=(self.q,))
		self._active = False
		
	def display(self):
		self.p.start()
		self._active = True
		
	def update(self, data):
		self.q.put(data)
		
	def active(self):
		active = self.p.is_alive()
		if not active and self._active:
			self.cleanup()
			self._active = False
		return active
	
	def _render(self, q):
		plt.style.use(["dark_background"])
		fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
		
		avgFitness = []
		avg10Fitness = []
		bestFitness = []
		avgScore = []
		bestScore = []
		std = []
		duration = []
		
		generations = 0
		
		while plt.get_fignums():
			if not q.empty():
				data = q.get()
				avgFitness.append(data["avgFitness"])
				avgScore.append(data["avgScore"])
				bestFitness.append(data["bestAvgFitness"])
				bestScore.append(data["sampleScore"])
				avg10Fitness.append(data["top10AvgFitness"])
				std.append(data["sampleStd"])
				duration.append(data["time"])
				generations += 1
			
			plt.suptitle("Evolution")
			
			ax1.clear()
			ax1.set_ylim(bottom=0, top=max(5, int(1.5*max(max(avgScore, default=0), max(bestScore, default=0), max(std, default=0)))))
			ax1.plot(avgScore, c="cyan", label="Avg Score")
			ax1.plot(std, c="green", ls="--", label="Best Sample Std")
			ax1.plot(bestScore, c="red", label="Best Sample Score")
			ax1.legend(loc="upper left")
			ax1.set_ylabel("Points")
			ax1.set_title("Scores")
			ax1.grid(alpha=0.25, ls="--")
			
			ax2.clear()
			ax2.set_ylim(bottom=0, top=max(5, int(1.5*max(bestFitness, default=0))))
			ax2.plot(avgFitness, c="cyan", label="Avg Fitness")
			ax2.plot(bestFitness, c="red", label="Best Fitness")
			ax2.plot(avg10Fitness, c="green", label="Top 10 Avg Fitness")
			ax2.legend(loc="upper left")
			ax2.set_ylabel("Fitness")
			ax2.set_title("Fitnesses")
			ax2.grid(alpha=0.25, ls="--")
			
			ax3.clear()
			ax3.set_xlim(left=0, right=max(5, int(1.15*generations)))
			ax3.set_ylim(bottom=0, top=max(5, int(1.5*max(duration, default=0))))
			ax3.plot(duration, c="magenta", label="Gen Duration")
			ax3.legend(loc="upper left")
			ax3.set_ylabel("Seconds")
			ax3.set_title("Time")
			ax3.grid(alpha=0.25, ls="--")
			
			ax3.set_xlabel("Generations")
			try:
				plt.pause(0.1)
			except:  # raises exception if user exits window while running
				pass
		q.close()
		
	def cleanup(self):
		if self._active:
			self.p.join()

	