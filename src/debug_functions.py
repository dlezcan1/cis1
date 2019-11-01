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
import re


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
    error_b = np.round( np.abs( ( b - b_fit ) ), 2 )
    
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
    c = np.random.randn(10)
    Y = np.poly1d(c)(X)
     
    #==========================================================================
    # for a 5th order Bernstein polynomial, the value seems to 
    # divergs to large values. 2nd order appears to be good here for
    # this fit seems to be good for 2. Carries away for greater than 2
    #==========================================================================
    coeffs, qmin, qmax = cr.undistort( X, Y, 5 )
    Y_fit = [cr.correctDistortion( coeffs, v, qmin, qmax ) for v in X]
    Y_fit = np.array( Y_fit )
    
    errors = np.abs( Y_fit - Y )  # / Y
    
    print( 20 * '=', "Debug 'undistort'", 20 * '=' )
    print('Coefficients of random polynomial 10th order polynomial')
    print(c)
    print()
    
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
    
# debug_correct_C

def debug_compute_emfiducial():
    print(25*'=', ' compute fiducial points in em coord system ', 25*'=')

    file_name_empivot = "../pa1-2_data/pa2-debug-a-empivot.txt"
    file_name_calreadings = "../pa1-2_data/pa2-debug-a-calreadings.txt"
    file_name_emfiducials = "../pa1-2_data/pa2-debug-a-em-fiducialss.txt"
    file_name_output1 = "../pa1-2_data/pa2-debug-a-output1.txt"
    print("reading em fiducial data...")
    _, t_post = Program2.improved_empivot_calib( file_name_empivot)
    coef, qmin, qmax = Program2.undistort_emfield( file_name_calreadings, file_name_output1, 5)
    #Program2.compute_fiducials_em(file_name_emfiducials, coef, qmin, qmax, t_post)

def debug_undistort_emfield():
    """Function to debug 'Program2.undistort_emfield' function"""
    filename_calreadings = '../pa1-2_data/pa2-debug-a-calreadings.txt'
    filename_output1 = '../pa1-2_data/pa2-debug-a-output1.txt'
    C_exp_data = open_files.open_output1( filename_output1 )['C_expected']
    calread = open_files.open_calreadings( filename_calreadings )
    
    coeffs, qmin, qmax = Program2.undistort_emfield(filename_calreadings, filename_output1, 5)
    
    # read in only the C_readings
    C_read = []
    for frame in calread.keys():
        C_read.append( calread[frame]['vec_c'] )
        
    # for
    
    # put data into large array
    C_expected = []
    for frame in C_exp_data.keys():
        C_expected.append( C_exp_data[frame] )
        
    # for
    
    C_read = np.array( C_read ).reshape( ( -1, 3 ) )
    C_expected = np.array( C_expected ).reshape( ( -1, 3 ) )
    
    C_read_undistorted = [cr.correctDistortion(coeffs, c_i, qmin, qmax) 
                          for c_i in C_read]
    C_read_undistorted = np.array(C_read_undistorted)
    
    error = np.abs((C_expected-C_read_undistorted)/C_expected)
    
    print('C_read\n',C_read)
    print()
    
    print('C_expected\n',C_expected)
    print('C_undistorted\n',C_read_undistorted)
    print()
    print('Rel. Error\n',error)
    print('Max Rel. Error:', np.max(error))
    print('Avg. Rel. Error:', np.average(error))
        
# debug_undistort_emfield


def debug_improved_empivot_calib():
    """Function to test the 'Program2.improved_empivot_calib' function."""
    empivot_filename = '../pa1-2_data/pa2-debug-a-empivot.txt'
    print( 'Processing file:', empivot_filename )
    t_G, t_post = Program2.improved_empivot_calib( empivot_filename )
    print( 't_G:\n', t_G )
    print( 't_post\n', t_post )
    
# debug_improved_empivot_calib


def debug_compute_Freg():
    """Function to test the 'Program2.compute_Freg' function"""
    filename_ctfiducials = '../pa1-2_data/pa2-debug-a-ct-fiducials.txt'
    filename_emfiducials = '../pa1-2_data/pa2-debug-a-em-fiducialss.txt'
    
    # extract file metadata
    file_pattern = r'pa2-(debug|unknown)-(.)-ct-fiducials.txt'
    file_fmt = '../pa1-2_data/pa2-{0}-{1}-{2}.txt'
    res_empivot = re.search( file_pattern, filename_ctfiducials )
    data_type, letter = res_empivot.groups()
    
    # generate related files
    filename_empivot = file_fmt.format( data_type, letter, 'empivot' )
    filename_calreadings = file_fmt.format( data_type, letter, 'calreadings' )
    filename_output1 = file_fmt.format( data_type, letter, 'output1' )
    fid_em = open_files.open_emfiducials( filename_emfiducials )
    
    # compute Freg and obtain b coords
    Freg = Program2.compute_Freg( filename_ctfiducials, filename_emfiducials )
    b = open_files.open_ctfiducials( filename_ctfiducials )
    
    # perform empivot calibration
    t_G, _ = Program2.improved_empivot_calib( filename_empivot )
    t_G_hom = np.append( t_G, 1 )  # homogeneous representation
    
    coeffs, qmin, qmax = Program2.undistort_emfield( filename_calreadings, filename_output1, 5 )
    
    # correct em_fiducial data
    fid_em_calibrated = {}
    for frame in fid_em.keys():
        coords = fid_em[frame]
        coords_calib = [cr.correctDistortion( coeffs, v, qmin, qmax ) for v in coords]
        fid_em_calibrated[frame] = np.array( coords_calib )
        
    # for
    
    # initializations for pose determination of EM point
    G_first = fid_em_calibrated['frame1']
    G_zero = np.mean( G_first, axis = 0 )
    g_j = G_first - G_zero
    zoom = np.ones( 3 )  # no zooming
    
    B_matrix = []  # where to contain the B_i values
    for frame in fid_em_calibrated.keys():
        # for each frame, compute transformation of F_G[k]
        G = fid_em_calibrated[frame]
        # frame transformation [g_j -> G]
        F_G = cr.point_cloud_reg( g_j, G )
        # homogeneous representation
        F_G = tf3e.affines.compose( F_G['Trans'],
                                    F_G['Rotation'], zoom )
        
        B_i = F_G.dot( t_G_hom )[:3]
        B_matrix.append( B_i )
        
    # for
    B_matrix = np.array( B_matrix )
    
    b_fit = [Freg.dot( np.append( B, 1 ) )[:3] for B in B_matrix]
    b_fit = np.array( b_fit )
    
    error = np.abs( ( b_fit - b ) / b )

    print('Freg\n',Freg)
    print()
    
    print( "b\n", b )
    print()
    
    print( 'b_fit\n', b_fit )
    print()
    
    print( 'Rel. Error\n', error )
    print('Max Rel. Error:', np.max(error))
    print('Avg. Rel. Error:', np.average(error))

# debug_compute_Freg

def debug_compute_test_points():
    file_name_emnav = "../pa1-2_data/pa2-debug-a-EM-nav.txt"
    Program2.compute_test_points(file_name_emnav)
    

    
if __name__ == '__main__':
#     debug_point_cloud()
    #debug_point_cloud_reg()
#     debug_calibration()
    #debug_undistort()
#    debug_correct_C()
    #debug_compute_emfiducial()
    
#     debug_undistort_emfield()
#     debug_improved_empivot_calib()
    #debug_compute_Freg()
    debug_compute_test_points()

# if
    
