'''
Created on Oct 23, 2019

@author: Dimitri Lezcano and Hyunwoo Song

This function provides several methods to debug our program
that we have built.
'''
import open_files, Program1, Program2  # @UnusedImport
import Calibration_Registration as cr
import transforms3d_extend as tf3e  # @UnusedImport
import numpy as np
import transformations
from scipy.interpolate import BPoly
from Calibration_Registration import generate_berntensor


def debug_point_cloud():
    """ Method to test the current point cloud registration 
        algorithm.
        
        @author: Dimitri Lezcano and Hyunwoo Song
        
    """
    print( 20 * '=', 'Functionality Check', 20 * '=' )
    file_a_calbody = '../pa1-2_data/pa1-debug-a-calbody.txt'
    calbody = open_files.open_calbody( file_a_calbody )
    print( "data type: ", calbody['vec_a'].dtype )
    # numpy float64
    # calbody = open_files.open_calbody_npfloat(file_a_calbody)
    
    print( "data type: ", calbody['vec_a'].dtype )
    print( "Calbody:\n", calbody['vec_a'] )
    
    file_a_calreadings = '../pa1-2_data/pa1-debug-a-calreadings.txt'
    calreadings = open_files.open_calreadings( file_a_calreadings )
    # numpy float 64
    # calreadings = open_files.open_calreadings_npfloat(file_a_calreadings)

    print( "data type: ", calreadings['frame1']['vec_a'].dtype )
    print( "Calreadings:\n", calreadings['frame1']['vec_a'] )
    
    F = cr.point_cloud_reg( calbody['vec_a'], calreadings['frame1']['vec_a'] )
    print( "F: \n", F )

    b_calc = np.dot( calbody['vec_a'], F['Rotation'] ) + F['Trans']
    print( "b_origin: \n", calreadings['frame1']['vec_a'] )
    print( "b_calc: \n", b_calc )
    print( "b_origin equals b_calc?", np.array_equal( calreadings['frame1']['vec_a'], b_calc ) )
    print()
    
    print( 25 * '=', 'TEST', 25 * '=' )
    R = transformations.rotation_matrix( np.pi / 2, [1, 0, 0] )[:3, :3]
    t = np.array( [1, 2, 3] )
    print( "R:\n", R )
    a = np.eye( 3 )
    b = R.dot( a ) + t
    
    F = cr.point_cloud_reg( a, b )
    print( "Point Cloud R:\n", F['Rotation'] )
    print()
    print( "t:\n", t )
    print( "Point cloud t:\n", F["Trans"] )
    print( "Rotation close:", str( np.allclose( R, F['Rotation'] ) ) )
    print( "Translations close:", str( np.allclose( t, F['Trans'] ) ) )
    
    print( "R.I + t:" )
    print( b )
    
    print( "R_pt.I + T:" )
    print( F["Rotation"].dot( a ) + F["Trans"] )
    
# _debug_point_cloud


def debug_point_cloud_reg():
    """ This method is to debug the 'point_cloud_reg' function 
        in another manner, using random quaternions  and vectors 
        to generate transformations.
    """
    
    q = np.random.randn( 4 )
    q = q / np.linalg.norm( q )
    
    R = tf3e.quaternions.quat2mat( q )
    
    p = np.random.randn( 3 )
    
    a = np.random.randn( 8, 3 )  # random a vectors
    
    b = np.array( [R.dot( ai ) + p for ai in a] )
    
    F = cr.point_cloud_reg( a, b )
    b_fit = np.array( [F['Rotation'].dot( ai ) + F['Trans'] for ai in a] )
    error_R = np.round( np.abs( R - F['Rotation'] ), 2 )
    error_p = np.round( np.abs( p - F['Trans'] ), 2 )
    error_b = np.round( np.abs( b - b_fit ), 2 )
    
    print( 25 * "*", 'Test "point_cloud_reg"', 25 * '*' )
    print( 'Random quaternion:', q )
    print( 'Associated Rotation Matrix:\n ', R )
    print( "Registered Rotation Matrix:\n", F['Rotation'] )
    print( "R.R^T\n", np.round( R.dot( R.T ), 2 ) )
    print()
    
    print( "Random Translation vector:\n", p )
    print( "Registered Translation vector:\n", F['Trans'] )
    print()
    
    print( 'Random points generated\n', a )
    print( 'Mapped points [R,p].a\n', b )
    print( "Fitted points:\n", b_fit )
    print()
    
    print( "|R - R_fit|\n", error_R )
    print()
    
    print( '|p - p_fit| \n', error_p )
    print()
    
    print( '|b - b_fit|\n', error_b )
    
# debug_point_cloud_reg


def debug_calibration():
    """ This method is to debug the pivot calibration method,
        'pointer_calibration'
    """

    print( 25 * "=", "Random pointer, post position", 25 * "=" )
    p_ptr = np.random.randn( 3 )
    p_post = np.random.randn( 3 )
    
    print( "p_ptr: \n", p_ptr )
    print( "p_post: \n", p_post )
    print()
    no_mat = 10
    print( "Random rotation matrix generated: ", no_mat )
    print()
    F_G = []
    for mIdx in range( no_mat ):
        tmp_rot = np.random.rand( 3, 3 )
        tmp_trans = p_post - tmp_rot.dot( p_ptr )
        F_tmp = tf3e.affines.compose( tmp_trans, tmp_rot, np.ones( 3 ) )
        F_G.append( F_tmp )
    
    print( 25 * "=", "Calibration functionality check", 25 * "=" )
    p_ptr_cali, p_post_cali = cr.pointer_calibration( F_G )
    print( "p_ptr_cali: \n", p_ptr_cali )
    print( "p_post_cali: \n", p_post_cali )
    print()

    print( 25 * "=", "Debug result", 25 * "=" )
    print( "pointer diff: ", p_ptr - p_ptr_cali )
    print( "equal? ", np.array_equal( p_ptr, p_ptr_cali ) )
    print( "post diff: ", p_post - p_post_cali )
    print( "equal? ", np.array_equal( p_post, p_post_cali ) )
          

def debug_undistort():
    """This function is to debug the 'undistort' function"""
    X = np.random.randn( 6, 3 )
#     qmin = np.min( X )
#     qmax = np.max( X )
#     tensor = generate_berntensor( X, qmin, qmax, 1 )
#     c = np.ones( ( 2 ** 3, 3 ) )
#     Y = tensor.dot(c)
    Y = X ** 2 + 2 * X + 1
     
    #==========================================================================
    # for a 5th order Bernstein polynomial, the value seems to 
    # divergs to large values. 2nd order appears to be good here for
    # this fit seems to be good for 2. Carries away for greater than 2
    #==========================================================================
    coeffs, qmin, qmax = cr.undistort( X, Y, 5 )
    Y_fit = [cr.correctDistortion(coeffs, v, qmin, qmax) for v in X]
    Y_fit = np.array( Y_fit )
    
    errors = np.abs( Y_fit - Y )  # / Y
    
    print( 20 * '=', "Debug 'undistort'", 20 * '=' )
    print( "X" )
    print( X )
    print( "X_normalized" )
    print( cr.scale_to_box( X, qmin, qmax )[0] )
    
    print( "Y" )
    print( Y )
    print( "Y_fit" )
    print( Y_fit )
    print()
    
    print( "|Y-Y_fit|/Y " )
    print( np.round( errors, 2 ) )
    
# debug_undistort

def debug_correct_C():
    print(25*"=", " debug_correct_C " , 25*"=")

    file_name_calreadings = "../pa1-2_data/pa2-debug-a-calreadings.txt"
    file_name_output1 = "../pa1-2_data/pa2-debug-a-output1.txt"
    # Read C_expected from output1
    print("reading C_expected...")
    C_exp_data = open_files.open_output1(file_name_output1)['C_expected']
    C_expected = []
    for frame in C_exp_data.keys():
        C_expected.append(C_exp_data[frame])

    #print("C_expected(first frame): \n", C_expected[0])
    #print()

    # Dewarping C
    print("dewarping C...")
    coef, qmin, qmax = Program2.undistort_emfield( file_name_calreadings, file_name_output1, 5)
    C_undistorted, outfile = Program2.correct_C(file_name_calreadings, coef, qmin, qmax)

    #print("C_undistorted(first frame): \n", C_undistorted[0])
    #print()

    # Check the error
    
    print("Dewarp correct? ", np.array_equal(C_expected, C_undistorted))
    if np.array_equal(C_expected, C_undistorted) is False:
        error = np.array([u-v for u, v in zip(C_expected, C_undistorted)])
        print("Error: \n", error)

    print(25*"=", " Debugging finished ", 25*"=")
    

if __name__ == '__main__':
#     debug_point_cloud_reg()
#     debug_calibration()
    #debug_undistort()
    debug_correct_C()
    
# if
    
