'''
Created on Oct 23, 2019

@author: dlezcan1
'''
import open_files, Program1  # @UnusedImport
import Calibration_Registration as cr
import transforms3d_extend as tf3e  # @UnusedImport
import numpy as np
import transformations

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
    q = np.random.randn(4)
    q = q/np.linalg.norm(q)
    
    R = tf3e.quaternions.quat2mat(q)
    
    p = np.random.randn(3)
    
    
    a = np.random.randn(8, 3) # random a vectors
    
    b = np.array([R.dot(ai) + p for ai in a])
    
    F = cr.point_cloud_reg(a, b)
    b_fit = np.array([F['Rotation'].dot(ai) + F['Trans'] for ai in a])
    error_R = np.round(np.abs(R-F['Rotation']),2)
    error_p = np.round(np.abs(p-F['Trans']),2)
    error_b = np.round(np.abs(b - b_fit),2)
    
    print(25*"*", 'Test "point_cloud_reg"', 25*'*')
    print('Random quaternion:', q)
    print('Associated Rotation Matrix:\n ', R)
    print("Registered Rotation Matrix:\n", F['Rotation'])
    print("R.R^T\n", np.round(R.dot(R.T),2))
    print()
    
    print("Random Translation vector:\n", p)
    print("Registered Translation vector:\n", F['Trans'])
    print()
    
    print('Random points generated\n', a)
    print('Mapped points [R,p].a\n', b)
    print("Fitted points:\n", b_fit)
    print()
    
    print("|R - R_fit|\n",error_R)
    print()
    
    print('|p - p_fit| \n', error_p)
    print()
    
    print('|b - b_fit|\n', error_b)
    
# debug_point_cloud_reg

def debug_calibration():
    print(25*"=", "Random pointer, post position", 25*"=")
    p_ptr = np.random.randn(3)
    p_post = np.random.randn(3)
    
    print("p_ptr: \n", p_ptr)
    print("p_post: \n", p_post)
    print()
    no_mat = 10
    print("Random rotation matrix generated: ", no_mat)
    print()
    F_G = []
    for mIdx in range(no_mat):
        tmp_rot = np.random.rand(3,3)
        tmp_trans = p_post - tmp_rot.dot(p_ptr)
        F_tmp = tf3e.affines.compose(tmp_trans, tmp_rot, np.ones(3))
        F_G.append(F_tmp)
    
    print(25*"=", "Calibration functionality check", 25*"=")
    p_ptr_cali, p_post_cali = cr.pointer_calibration(F_G)
    print("p_ptr_cali: \n", p_ptr_cali)
    print("p_post_cali: \n", p_post_cali)
    print()

    print(25*"=", "Debug result", 25*"=")
    print("pointer diff: ", p_ptr - p_ptr_cali)
    print("equal? ", np.array_equal(p_ptr, p_ptr_cali))
    print("post diff: ", p_post - p_post_cali)
    print("equal? ", np.array_equal(p_post, p_post_cali))
          

if __name__ == '__main__':
    debug_point_cloud_reg()
    debug_calibration()
    
# if
    
