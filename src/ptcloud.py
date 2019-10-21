""" This file is about 3d point set to 3d point set registration algorithm"""

import sys
import numpy as np

points_a = open("pa1-debug-a-calbody.txt", "r")
#print(points_a.read())

while True:
    line = points_a.readlines()
    if not line: break
    #print(line)
points_a.close()

tmp = np.genfromtxt('pa1-debug-a-calbody.txt', encoding='ascii', dtype=None, skip_header=1)
print(tmp)
print(tmp[1][0])
print(tmp[1][2])
