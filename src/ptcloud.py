""" This file is about 3d point set to 3d point set registration algorithm"""
import sys
import numpy as np
import open_files

file_a = 'pa1-debug-a-'

calbody = open_files.open_calbody(file_a)
print(calbody)

calreadings = open_files.open_calreadings(file_a)
print(calreadings)

