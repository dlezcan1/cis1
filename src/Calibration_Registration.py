'''
Created on Oct 17, 2019

@author: Dimitri Lezcano and Hyunwoo Song

@name: Calibration_Registration

@summary: This module is to provide python functions of calibration and 
          registration for computer-integrated surgical interventions.
'''

import transforms3d_extend
import numpy as np
from scipy.interpolate import BPoly


def point_cloud_reg( a, b ):
    """ This function read the two coordinate systems and
        calculate the point-to-point registration function.
        This algorithm is explained in class and implemented as taught.
        The relationship between a, b, and the result ("F") is
            b = F a
        Basically, this function compute the H matrix and calculate the
        unit quaternion at the largest eigen value of H.
        Then it uses quaternion to calculate the rotation matrix.
        Translation vector is calculated by 
            trans = b - R dot a 

        @author: Hyunwoo Song

        @param a: the input, numpy array where the vectors are the rows of the
                  matrix
                  
        @param b: the corresponding output, numpy array where the vectors are
                  the rows of the matrix

        @return: F which is a dictionary consist of 'Ratation' as a rotation
                matrix and 'Trans' as a translational vector
    
    """
    mean_a = np.mean( a, axis = 0 )
    mean_b = np.mean( b, axis = 0 )
    
    # Compute for mean and subtract from a, b, respectively
    a_hat = a - mean_a
    b_hat = b - mean_b

    # Compute for H
    mult = np.multiply( a_hat, b_hat )
    ab_xx = np.sum( mult[:, 0] )
    ab_yy = np.sum( mult[:, 1] )    
    ab_zz = np.sum( mult[:, 2] )

    ab_xy = np.sum( np.multiply( a_hat[:, 0], b_hat[:, 1] ) )
    ab_xz = np.sum( np.multiply( a_hat[:, 0], b_hat[:, 2] ) )
    ab_yx = np.sum( np.multiply( a_hat[:, 1], b_hat[:, 0] ) )
    ab_yz = np.sum( np.multiply( a_hat[:, 1], b_hat[:, 2] ) )
    ab_zx = np.sum( np.multiply( a_hat[:, 2], b_hat[:, 0] ) )
    ab_zy = np.sum( np.multiply( a_hat[:, 2], b_hat[:, 1] ) )
    
    H = np.array( [[ab_xx, ab_xy, ab_xz], [ab_yx, ab_yy, ab_yz],
                   [ab_zx, ab_zy, ab_zz]] )

    # Compute G
    H_trace = np.trace( H )
    Delta_trans = np.array( [H[1, 2] - H[2, 1], H[2, 0] - H[0, 2], H[0, 1] - H[1, 0]] )
    Delta = Delta_trans.reshape( ( -1, 1 ) )

    G = np.vstack( ( np.hstack( ( H_trace, Delta_trans ) ),
                      np.hstack( ( Delta, H + H.T - H_trace * np.eye( 3 ) ) ) ) )
    
    a_eigenVal, m_eigenVec = np.linalg.eig( G )
    
    # unit quaternion
    q = m_eigenVec[np.argmax( a_eigenVal )]
    
    # Calculate R using unit quaternion
    R_00 = q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2
    R_01 = 2 * ( q[1] * q[2] - q[0] * q[3] )
    R_02 = 2 * ( q[1] * q[3] + q[0] * q[2] )
    R_10 = 2 * ( q[1] * q[2] + q[0] * q[3] )
    R_11 = q[0] ** 2 - q[1] ** 2 + q[2] ** 2 - q[3] ** 2
    R_12 = 2 * ( q[2] * q[3] - q[0] * q[1] )
    R_20 = 2 * ( q[1] * q[3] - q[0] * q[2] )
    R_21 = 2 * ( q[2] * q[3] + q[0] * q[1] )
    R_22 = q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2

    R = np.array( [[R_00, R_01, R_02], [R_10, R_11, R_12], [R_20, R_21, R_22]] )
    R_pack = transforms3d_extend.quaternions.quat2mat( q )

        # Calculate translation

    p = mean_b - R.dot( mean_a ) 
    
    F = {'Rotation': R, 'Trans': p}
    return F

# point_cloud_reg


def pointer_calibration( transformation_list: list ):
    """Function that determines the parameters of the pointer given a pivot 
    calibration data for the pointer. 
    This method is implemented as in class. 
    The least squares function used here is proved by 'numpy.linalg.lstsq'.
    
    Solves the least squares problem of:
      ...       ...                ...
    { R_j      -I  }{ p_ptr  } = { -p_j }
      ...      ...    p_post       ...
    where:
    -> (R_j,p_j) is the j-th transformation of the pivot
    -> p_ptr     is the vector of the pointer posistion relative to the tracker
    -> p_post    is the vector position of the post. 
    
    @author: Dimitri Lezcano
      
    @param transformation_list: A list of the different transformation matrices 
                                from the pivot calibration
    
    @return: [p_ptr, p_post] where they are both 3-D vectors. 
    """
    
    coeffs = np.array( [] )
    translations = np.array( [] )
    for i, transform in enumerate( transformation_list ):
        # split the transformations into their base matrices and vectors
        # zoom and shear assumed to be ones and zeros, respectively
        p, R, _, _ = transforms3d_extend.affines.decompose44( transform ) 
        C = np.hstack( ( R, -np.eye( 3 ) ) )
        if i == 0:  # instantiate the sections
            coeffs = C
            translations = -p
        
        # if
        
        else:  # add to the list
            coeffs = np.vstack( ( coeffs, C ) )
            translations = np.hstack( ( translations, -p ) )
            
        # else
    # for
        
    lst_sqr_soln, _, _, _ = np.linalg.lstsq( coeffs, translations, None )
    # p_ptr  is indexed 0-2
    # p_post is indexed 3-5
    
    return [lst_sqr_soln[:3], lst_sqr_soln[3:]]

# pointer_calibration


def generate_Bpoly_basis( N: int ):
    """This function is to generate a basis of the bernstein polynomial basis
       
    
       @author: Dimitri Lezcano
       
       @param N: an integer representing the highest order or the 
                 Bernstein polynomial.
    
       @return:  A list of Bernstein polynomial objects of size N, that will 
                 individually be B_0,n, B_1,n, ..., B_n,n
                
    """
    zeros = np.zeros( N + 1 )
    
    x_break = [0, 1]
    basis = []
    
    for i in range( N ):
        c = np.copy( zeros )
        c[i] = 1
        c = c.reshape( ( -1, 1 ) )
        basis.append( BPoly( c, x_break ) )
        
    # for 
    
    return basis

# generate_Bpoly


def scale_to_box( X: np.ndarray ):
    """A Function to scale an input array of vectors and return 
       the scaled version from 0 to 1
       
       @author Dimitri Lezcano
       
       @parap X: a numpy array where the rows are the corresponding vectors
                 to be scaled.
     
       @return: X', the scaled vectors given from the function.
       
    """
    qmin = np.min( X, 0 )  # minimum vector for scaling
    qmax = np.max( X, 0 )  # maximum vector for scaling
    div = np.linalg.norm( qmax - qmin )
    
    X_prime = ( X - qmin ) / div  # normalized input
    
    return X_prime
    
# scale_to_box


def undistort( X: np.ndarray, Y :np.ndarray, order:int ):
    """Function to undistort a calibration data set using Bernstein polynomials.
    
       Solving the least squares problem of type:
          ...        ...        ...       c_0       ...
       ( B_0,N(x_j)  ...    B_N,N(x_j) )( ... ) = ( p_j )
          ...        ...        ...       c_N       ...
          
       @author: Dimitri Lezcano
       
       @param X: The input parameters to be fit to Y
       
       @param Y: The output parameters to be fit from X
       
       @param order: The highest order that would like to be fitted of the 
                     Bernstein polynomial
       
       @return: A Bernstein Polynomial object with the fitted coefficients
    
    """
    bern_basis = generate_Bpoly_basis( order )
    X_prime = scale_to_box( X )  # "normalized" vectors
    
    bern_Matrix = None
    for B in bern_basis:
        if isinstance( bern_Matrix, type( None ) ):  # instantiation
            bern_Matrix = B( X_prime )
        
        else:  # horizontally stack
            bern_Matrix = np.hstack( ( bern_Matrix, B( X_prime ) ) )
    
    # for
    
    lstsq_soln, _, _ = np.linalg.lstsq( bern_Matrix, Y, None )
    
    coeffs = lstsq_soln.reshape( ( -1, 1 ) )
    
    return BPoly( coeffs, [0, 1] )
    
# undistort
