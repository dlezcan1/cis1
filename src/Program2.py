'''
Created on Oct 26, 2019

@author: Hyunwoo Song and Dimitri Lezcano

@summary: This module is to answer the Programming Assignment 1's specific
          questions. 
'''

import Calibration_Registration as cr
import numpy as np
import glob
import open_files
import transforms3d_extend as tf3e 
import re
from Program1 import compute_Cexpected


def improved_empivot_calib( filename_empivot: str ):
    """This function is to provide an imporved pivot calibration of the EM
       probe using the undistort function provided in Calibration_Registration
       to remove distortion in the EM field.
       
       Here the EM field will be undistorted using the C_expected values
       and then the EM pivot calibration will be performed. using the
       undistorted values.
       
       @author: 
              
       @param filename_empivot:  a string representing the data file for the 
                                 EM pivot tracking to be read in.
       
       @param filename_optpivot: a string representing the data file for the
                                 optical pivot tracking to be read in
                                 MAY NOT BE NECESSARY
                                 
       @return: 
       
    """
    # locate associated calreadings and output file for this file.
    file_pattern = r'pa2-(debug|unknown)-(.)-empivot.txt'
    file_fmt = '../pa1-2_data/pa2-{0}-{1}-{2}.txt'
    res_empivot = re.search( file_pattern, filename_empivot )
    data_type, letter = res_empivot.groups()
    filename_calreadings = file_fmt.format( data_type, letter, 'calreadings' )
    filename_output1 = file_fmt.format( data_type, letter, 'output1' )
    
    # find the distortion coefficients
    coeffs, qmin, qmax = undistort_emfield( filename_calreadings, filename_output1, 2 )
    
    # read in EM pivot data
    empivot = open_files.open_empivot( filename_empivot )
    
    # correct empivot data
    empivot_calibrated = {}
    for frame in empivot.keys():
        coords = empivot[frame]
        coords_calib = [cr.correctDistortion( coeffs, v, qmin, qmax ) for v in coords]
        empivot_calibrated[frame] = np.array(coords_calib)
        
    # for
    
    # perform the EM pivot calibration with the calibrated data
    G_first = empivot_calibrated['frame1']
    G_zero = np.mean(G_first, axis = 0)
    g_j = G_first - G_zero
    
    Trans_empivot = []
    zoom = np.ones( 3 )  # for frame composition
    for frame in empivot_calibrated.keys():
        ################## b ################
        # for each frame, compute transformation of F_G[k]
        G = empivot_calibrated[frame]
        # frame transformation [g_j -> G]
        F_G = cr.point_cloud_reg( g_j, G )
        # homogenous representation
        F_G = tf3e.affines.compose( F_G['Trans'],
                                                  F_G['Rotation'], zoom )
        Trans_empivot.append( F_G )

    ############## c ################
    # pivot calibration
    t_G, p_post = cr.pointer_calibration( Trans_empivot )
    
    return t_G, p_post

# improved_empivot_calib


def undistort_emfield( filename_calreadings, filename_output1: str, order_fit: int ):
    """This function is to provide an imporved pivot calibration of the EM
       probe using the undistort function provided in Calibration_Registration
       to remove distortion in the EM field.
       
       Here the EM field will be undistorted using the C_expected values
       and then the EM pivot calibration will be performed. using the
       undistorted values.
       
       @author: Dimitri Lezcano
              
       @param filename_calreadings:  a string representing the data file for the
                                     calibration body read data to be read in
       
       @param filename_output1:  a string representing the data file for the
                                 output1 C_expected to be read in
       
       @param order_fit:         an integer representing which order of the 
                                 Bernstein polynomial fitting that will be used.

                                 
       @return: coefficent matrix for distortion correction of EM field
                qmin, the minimum value for scaling
                qmax, the maximum value for scaling
       
    """
    # read in files
    C_exp_data = open_files.open_output1( filename_output1 )['C_expected']
    calread = open_files.open_calreadings( filename_calreadings )
    
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
    
    coeffs, qmin, qmax = cr.undistort( C_read, C_expected, order_fit )
    
    return coeffs, qmin, qmax
    
# undistort_emfield

def correct_C(filename_calreadings : str, coef : np.ndarray, qmin, qmax):
    """ This functions corrects the C values from calreading txt file with respect to
        EM tracker base coordinate system.

        @author: Hyunwoo Song

        @param filename_em_fiducials: string of the filename to be read

        @return: position(x,y,z) of the fiducial points
    """

    name_pattern = r'pa(.)-(debug|unknown)-(.)-calreadings.txt'
    res_calreading = re.search(name_pattern, filename_calreadings)
    assign_num, data_type, letter = res_calreading.groups()
    outfile = "../pa{0}_results/pa{0}-{1}-{2}-output{0}.txt".format(assign_num,
                                                                    data_type,
                                                                    letter)

    Cal_readings = open_files.open_calreadings(filename_calreadings)
    C_undistorted = []
    for idx, frames in enumerate(Cal_readings):
        print('frame %d/%d' %(idx+1, len(Cal_readings.keys())))
        C_distorted = Cal_readings[frames]['vec_c']
        #print("C_distorted shape: ", np.shape(C_distorted))
        #print("C_distorted: \n", C_distorted)

        retval = np.array([cr.correctDistortion(coef, C_tmp, qmin, qmax) for C_tmp in C_distorted])
        #print("retval shape: ", np.shape(retval))
        #print("retval : \n",retval)
    
        C_undistorted.append( retval )

    #print("C_undistorted \n", C_undistorted)
    #print("C_undistorted shape: ", np.shape(C_undistorted))

    # write corrected C to output1.txt 
    with open(outfile, 'w+') as writestream:
        outname = outfile.split( '/' )[-1]
        writestream.write( "{0}, {1}, {2}\n".format(len( Cal_readings['frame1']['vec_c'] ),
                                                    len( Cal_readings.keys() ),
                                                    outname ) ) # first line
        #write the undistorted C
        for frame in C_undistorted:
            for c in frame:
                writestream.write( "{0:.2f}, {1:.2f}, {2:.2f} \n".format(*c))

    print("File '{}' written.".format(outfile))

    return [C_undistorted, outfile]
# correct_C

if __name__ == '__main__':

    pass
