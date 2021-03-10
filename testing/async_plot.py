"""
Testing graph process.
"""

import os
from time import sleep, time
import numpy as np
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
	
def graph(q, points):
	print("Starting child", os.getpid())
	fig, axes = plt.subplots(2)
	
	while plt.get_fignums():
		if not q.empty():
			data = q.get()
			points.append(data)
			
		axes[0].clear()
		axes[1].clear()
		axes[0].plot(*list(zip(*points)))
		axes[1].plot(*list(zip(*[(point[0], np.cos(1 + 1/(1 + point[1]))) for point in points])))
		plt.pause(0.001)
	q.close()
	print("Exiting!")

def f(x):
	return np.sin(x**2)
	
def main():
	print("Starting parent", os.getpid())
	q = Queue()
	p = Process(target=graph, args=(q, [(x, f(x)) for x in range(5)]))
	p.start()
	
	for x in range(10, 1500):
		timer = time()
		q.put((x, f(x)))
		print("Sending", x)
		if not p.is_alive():
			#q.close()
			break

	print("Calculated...")
	p.join()
	print("Done!")

if __name__ == "__main__":
	main()
	