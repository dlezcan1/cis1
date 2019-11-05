'''
Created on Oct 21, 2019

@author: Dimitri Lezcano and Hyunwoo Song

@summary: This module is to extend the functionality of the transforms3d project
'''

from transforms3d import *  # @UnusedWildImport, meant for easy importing
import numpy as np


def inverse_transform44( matrix: np.ndarray ):
    """This method returns the transformation inverse given a 4x4 homogenous transform
       matrix, assuming there is no zoom or shear. This method is used to reduce the 
       computation time of inverting rigid frame transformations.
       
       @param matrix: a numpy, 4x4 matrix representing a rigid transformation
       
       @return: a 4x4 matrix representing the inverse of the input rigid transformation
    
    """
    if matrix.shape != ( 4, 4 ):
        raise IndexError( "The size of 'matrix' is not 4x4." )
    
    T, R, Z, S = affines.decompose44( matrix )
    Rinv = np.transpose( R )
    Tinv = -Rinv.dot( T )
    
    return affines.compose( Tinv, Rinv, Z, S )

# inverse_transform44


def small_error_frame( alpha: np.ndarray, epsilon: np.ndarray ):
    """This function is to return a 4x4 homogenous transformation matrix
       representing a small error as a frame
       
       @author: Dimitri Lezcano
       
       @param alpha: the rotation vector
       
       @param epsilon: the translation vector
       
       @return: a 4x4 homogenous vector of the form
                [ I + sk(alpha)  epsilon ]
                [     0         1    ]
                
    """
    
    dR = np.eye( 3 ) + skew( alpha )
    top = np.hstack( ( dR, epsilon.reshape( ( -1, 1 ) ) ) )
    bottom = np.array( [0, 0, 0, 1] )
    dF = np.vstack( ( top, bottom ) )
    
    return dF

# small_error_frame
    

def skew( vector: np.ndarray ):
    """ A function to return the skew-symmetric of the vector.
        
        @param vector: A 3-D vector 
        
        @returns: a 3x3 skew-symmetric matrix given the 3-D vector
    """
    vector = vector.astype( np.float )
    
    return np.array( [[0, -vector[2], vector[1]],
                      [vector[2], 0, -vector[0]],
                      [-vector[1], vector[0], 0]] )
    
# skew


if __name__ == "__main__":
    # check invserse
    q = np.random.randn( 4 )
    q = q / np.linalg.norm( q )
    R = quaternions.quat2mat( q )
    
    p = np.random.randn( 3 )
        
    F = affines.compose( p, R, np.ones( 3 ) )
    invF = inverse_transform44( F )
    print( F )
    print()
    print( invF )
    print()
    print( F.dot( invF ) )
    print( "F.invF close to I?: {}".format( str( np.allclose( F.dot( invF ), np.eye( 4 ) ) ) ) )
