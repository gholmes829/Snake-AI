"""
Tests time efficiency between methods of having base 36 numbers.
"""

import numpy as np
from time import time

def addOne(curr):  # won't cause overflow, might be slower
	base36 = list(curr)[::-1]  # d_0, d_1, ..., d_n
	
	carry = 1
	for i, digit in enumerate(base36):
		base36[i] = np.base_repr(int(base36[i], 36) + carry, 36)
		if len(base36[i]) > 1:
			carry = 1
			base36[i] = "0"
		else:
			carry = 0
			break
	if carry:
		base36.append("1")
			
			
	return "".join(base36[::-1])

def test1(test, trials):
	timer = time()
	for _ in range(trials):
		result = addOne(test)
	elapsed = time() - timer
	return elapsed, result
	
def test2(test, trials):  # could cause overflow if numbers get much larger than "ZZZZZ" base 36
	timer = time()
	for _ in range(trials):
		result = np.base_repr(int(test, 36) + 1, 36)
	elapsed = time() - timer
	return elapsed, result
	
def main():
	test = "1Z49ZZ"
	trials = 10000
	time1, result1 = test1(test, trials)
	time2, result2 = test2(test, trials)
	print(result1, time1)
	print(result2, time2)
	
if __name__ == "__main__":
	main()