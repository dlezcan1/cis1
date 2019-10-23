'''
Created on Oct 23, 2019

@author: dlezcan1
'''
import open_files, Program1  # @UnusedImport
import Calibration_Registration as cr
import transforms3d_extend as tf3e  # @UnusedImport
import numpy as np


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


if __name__ == '__main__':
    debug_point_cloud_reg()
    
# if
    