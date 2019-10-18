'''
Created on Oct 17, 2019

@author: Dimitri Lezcano, Hyunwoo Song

This module is to provide python functions of calibration and registration for computer-
integrated surgical interventions.
'''

import transforms3d
import numpy as np


def pointer_calibration(transformation_list: list):
    """Function that determines the parameters of the pointer given a pivot calibration
    data for the pointer. 
    I plan to use the linear least squares solver provided by numpy
    
    @param transformation_list: A list of the different transformation matrices from the 
                                pivot calibration
    
    @return: Returns both 
    
    """
    rotations = []
    translations = []
    for transform in transformation_list:
        #split the transformations into their base matrices and vectors
        R, _, p = transforms3d.affines.decompose44(transform) #leave out the zoom, assumed to be 1's
        
        
        
        