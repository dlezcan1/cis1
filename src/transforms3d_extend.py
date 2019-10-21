'''
Created on Oct 21, 2019

@author: Dimitri Lezcano and Hyunwoo Song

@summary: This module is to extend the functionality of the transforms3d project
'''

from transforms3d import *  # @UnusedWildImport, meant for easy importing
import numpy as np


def inverse_transform44(matrix: np.ndarray):
    """This method returns the transformation inverse given a 4x4 homogenous transform
       matrix, assuming there is no zoom or shear. This method is used to reduce the 
       computation time of inverting rigid frame transformations.
       
       @param matrix: a numpy, 4x4 matrix representing a rigid transformation
       
       @return: a 4x4 matrix representing the inverse of the input rigid transformation
    
    """
    print(matrix.shape)
    if matrix.shape != (4,4):
        raise IndexError("The size of 'matrix' is not 4x4.")
    
    T, R, Z, S = affines.decompose44(matrix)
    Rinv = np.transpose(R)
    Tinv = -Rinv.dot(T)
    
    return affines.compose(Tinv, Rinv, Z, S)

#inverse_transform44
