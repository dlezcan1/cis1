'''
Created on Oct 21, 2019

@author: Hyunwoo Song and Dimitri Lezcano

@summary: This module is to answer the Programming Assignment 1's specific
          questions. 
'''
import transforms3d_extend
import numpy as np
import glob
import open_files
import Calibration_Registration
import re


def compute_Cexpected( filename_calbody: str, filename_calreading: str ):
    """ This function is to undistort the calibration object viewed from 
        the EM tracker using the optical tracker as a ground truth to the
        position of the calibration object. This is a function for problem 4
    
        @param filename_calbody:    takes a string of the file name for 
                                    the calibration body
        @param filename_calreading: takes a string of the file name for 
                                    the calibration readings
        
        @return: C_j, the calibrated, expected values for the em-tracker 
                 position of the calibration body for each of the f
    
    """
    # attain the metaadta from filename
    name_pattern = r'pa(.)-(debug|unknown)-(.)-calbody.txt'
    res_calbody = re.search( name_pattern, filename_calbody )
    assign_num, data_type, letter = res_calbody.groups()
    outfile = "../pa{0}_results/pa{0}-{1}-{2}-output{0}.txt".format( assign_num,
                                                                    data_type,
                                                                    letter )
    
    calbody = open_files.open_calbody( filename_calbody )
    calib_data = open_files.open_calreadings( filename_calreading )
    
    frames = calbody.keys()
    zoom = np.ones( 3 )  # for frame composition
    
    C_expected_frames = []
    # start to iterate over the frames getting the required points
    for frame in frames: 
        ###################### part a ###################### 
        d_coords = calbody[frame]['vec_d']
        D_coords = calib_data[frame]['vec_d']
        
        # frame for d -> D
        F_D = Calibration_Registration.point_cloud_reg( d_coords, D_coords )
        # homogenous representation  
        F_D = transforms3d_extend.affines.compose( F_D['Trans'],
                                                  F_D['Rotation'], zoom )   
        
        ###################### part b ###################### 
        a_coords = calbody[frame]['vec_a']
        A_coords = calib_data[frame]['vec_a']
        
        # frame a -> A
        F_A = Calibration_Registration.point_cloud_reg( a_coords, A_coords )     
        # homogenous representation     
        F_A = transforms3d_extend.affines.compose( F_A['Trans'],
                                                  F_A['Rotation'], zoom )  
        
        ###################### part c ###################### 
        c_coords = [np.append( c, 1 ) for c in calbody[frame]['vec_c']]
        F_C = transforms3d_extend.inverse_transform44( F_D ) * F_A 
        
        C_expected = [F_C * c for c in c_coords]
        
        C_expected_frames.append( C_expected )

    # for

        
    ###################### part d ###################### 
    # write the output file for C_expected
    with open( outfile, 'w+' ) as writestream:
        writestream.write("{0}, {1}, {2}\n".format(len(calbody['frame1'],
                                                   len(frames),
                                                   outfile))) # first line
        
        # write the C_expected values 
        for C_expected in C_expected_frames:
            for c in C_expected:
                writestream.write("{0:.2f}, {1:.2f}, {2:.2f}\n".format(*c))
        
    # with
    
    return C_expected_frames
    
# compute_Cexpected


if __name__ == '__main__':
    pass
