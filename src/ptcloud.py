""" This file is about 3d point set to 3d point set registration algorithm"""
import sys
import numpy as np
import open_files

def point_cloud_reg(a, b):
    """ Rotation matrix -> Quaternion method for R
        Based on p13, "Rigid 3D 3D Calculations"

        Translation vector
    """
    mean_a = np.mean(a, axis=0)
    mean_b = np.mean(b, axis=0)

    # Compute for mean and subtract from a, b, respectively
    a_hat = a - mean_a
    b_hat = b - mean_b

    # Compute for H
    mult = np.multiply(a_hat, b_hat)
    ab_xx = np.sum(mult[:, 0])
    ab_yy = np.sum(mult[:, 1])    
    ab_zz = np.sum(mult[:, 2])

    ab_xy = np.sum(np.multiply(a_hat[:,0], b_hat[:,1]))
    ab_xz = np.sum(np.multiply(a_hat[:,0], b_hat[:,2]))
    ab_yx = np.sum(np.multiply(a_hat[:,1], b_hat[:,0]))
    ab_yz = np.sum(np.multiply(a_hat[:,1], b_hat[:,2]))
    ab_zx = np.sum(np.multiply(a_hat[:,2], b_hat[:,0]))
    ab_zy = np.sum(np.multiply(a_hat[:,2], b_hat[:,1]))
    
    H = np.array([[ab_xx, ab_xy, ab_xz],[ab_yx, ab_yy, ab_yz], [ab_zx, ab_zy, ab_zz]])

    # Compute G
    H_trace = np.trace(H)
    Delta_trans = np.array([H[1,2]-H[2,1], H[2,0]-H[0,2], H[0,1]-H[1,0]])
    Delta = Delta_trans.reshape((-1,1))

    G = np.vstack((np.hstack((H_trace, Delta_trans)), np.hstack((Delta, H+H.T-H_trace*np.eye(3)))))
    
    a_eigenVal, m_eigenVec = np.linalg.eig(G)
    
    # unit quaternion
    q = m_eigenVec[np.argmax(a_eigenVal)]
    
    # Calculate R using unit quaternion
    R_00 = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
    R_01 = 2*(q[1]*q[2] - q[0]*q[3])
    R_02 = 2*(q[1]*q[3] + q[0]*q[2])
    R_10 = 2*(q[1]*q[2] + q[0]*q[3])
    R_11 = q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2
    R_12 = 2*(q[2]*q[3] - q[0]*q[1])
    R_20 = 2*(q[1]*q[3] - q[0]*q[2])
    R_21 = 2*(q[2]*q[3] + q[0]*q[1])
    R_22 = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2

    R = np.array([[R_00, R_01, R_02],[R_10, R_11, R_12], [R_20, R_21, R_22]])

    # Calculate translation
    p = mean_b - np.multiply(R, mean_a)

    F = {'Rotation': R, 'Trans': p}
    return F


file_a = 'pa1-debug-a-'

calbody = open_files.open_calbody(file_a)
print(calbody['vec_a'])

calreadings = open_files.open_calreadings(file_a)
print(calreadings['frame1']['vec_a'])

F = point_cloud_reg(calbody['vec_a'], calreadings['frame1']['vec_a'])
print(F)
