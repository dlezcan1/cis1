'''
Created on Oct 17, 2019

@author: Dimitri Lezcano, Hyunwoo Song

This module is to provide python functions of calibration and registration for computer-
integrated surgical interventions.
'''

import transforms3d
import numpy as np
import os
import glob

def pointer_calibration(transformation_list: list):
    """Function that determines the parameters of the pointer given a pivot calibration
    data for the pointer. 
    I plan to use the linear least squares solver provided by numpy
    
    Solves the least squares problem of:
    ...        ...                  ...
    { $$R_j$$    -I  }{ p_ptr  } = { -p_j }
      ...        ...    p_post        ...
    where:
    -> (R_j,p_j) is the j-th transformation of the pivot
    -> p_ptr     is the vector of the pointer posistion relative to the tracker
    -> p_post    is the vector position of the post. 
      
    @param transformation_list: A list of the different transformation matrices from the 
                                pivot calibration
    
    @return: [p_ptr, p_post] where they are both 3-D vectors. 
    
    
    """
    coeffs = np.array([])
    translations = np.array([])
    for i, transform in enumerate(transformation_list):
        #split the transformations into their base matrices and vectors
        #zoom and shear assumed to be ones and zeros, respectively
        p, R, _, _ = transforms3d.affines.decompose44(transform) 
        C = np.hstack((R,-np.eye(3)))
        if i == 0: #instantiate the sections
            coeffs= C
            translations = -p
        
        #if
        
        else: #add to the list
            coeffs = np.vstack((coeffs,C))
            translations = np.hstack((translations,-p))
            
        #else
    #for
        
    lst_sqr_soln = np.linalg.lstsq(coeffs, translations, None)
    #p_ptr  is indexed 0-2
    #p_post is indexed 3-5
    
    return [lst_sqr_soln[:3],lst_sqr_soln[3:]]

#pointer_calibration

def _debug_pivot_calib():
        for debug in glob.glob('..\\pa1-2_data\\*debug*empivot.txt'):
            with open(debug) as file:
                lines = file.read().split('\n')
                
                #extract data from first line
                N_marks, N_frames, fname = lines[0].split(',')
                
                
            
#_debug_pivot_calib

if __name__=='__main__':
    _debug_pivot_calib()