'''
Created on Oct 21, 2019

@author: Hyunwoo Song and Dimitri Lezcano
, 0\n" as these will serve as place-holders for the post position
        not calculated in this function.
        
        i.e. (File name)
        filename_calbody    = pa1-unknown-c-calbody.txt
        filename_calreading = pa1-unknown-c-calreadings.txt
        output filename     = pa1-unknown-c-output1.txt

@summary: This module is to answer the Programming Assignment 1's specific
          questions. 
'''
import transforms3d_extend
import numpy as np
import glob
import open_files
import Calibration_Registration
import re


def _determine_FD( d_coords, D_coords ):
    """This function is to determine the FD frame and return the average FD
       FD frame.
       
       @author: Dimitri Lezcano
       
       @param d_coords: the coordinates of the pointer
       
       @param D_coords: the coordinates measured by the EM tracker
       
       @return: The averaged frame FD.
    """
    frames = D_coords.keys()
    
    FD_list = []
    for frame in frame:
        # frame for d -> D
        F_D = Calibration_Registration.point_cloud_reg( d_coords, D_coords )
        # homogeneous representation  
        F_D = transforms3d_extend.affines.compose( F_D['Trans'],
                                                  F_D['Rotation'], zoom )   
        FD_list.append( F_D )
        
    # for
    
    return np.mean( FD_list, 0 )
    
# _determine_FD


# Compute_Cexpected (Prob4)
def compute_Cexpected( filename_calbody: str, filename_calreading: str ):
    """ This function is to undistort the calibration object viewed from 
        the EM tracker using the optical tracker as a ground truth to the
        position of the calibration object. This is a function for problem 4.
        
        This function will write a file, of the same file format as 
        the output1 files provided, with the 2nd and third rows as 
        "0, 0
        @author: Dimitri Lezcano    
    
        @param filename_calbody:    takes a string of the file name for 
                                    the calibration body
        @param filename_calreading: takes a string of the file name for 
                                    the calibration readings
        
        
        @return: C_j, the calibrated, expected values for the em-tracker 
                 position of the calibration body for each of the frames
                 
        @return: the output filename from the saving process.
    """
    # attain the metadata from filename
    name_pattern = r'pa(.)-(debug|unknown)-(.)-calbody.txt'
    res_calbody = re.search( name_pattern, filename_calbody )
    assign_num, data_type, letter = res_calbody.groups()
    outfile = "../pa{0}_results/pa{0}-{1}-{2}-output{0}.txt".format( assign_num,
                                                                    data_type,
                                                                    letter )
    
    calbody = open_files.open_calbody( filename_calbody )
    calib_data = open_files.open_calreadings( filename_calreading )
    
    frames = calib_data.keys()
    zoom = np.ones( 3 )  # for frame composition
    
    # get the calibration body coordinates
    d_coords = calbody['vec_d']
    a_coords = calbody['vec_a']
    c_coords = [np.append( c, 1 ) for c in calbody['vec_c']]  # homogeneous rep.
    
    C_expected_frames = []
    # start to iterate over the frames getting the required points
    for frame in frames: 
        ###################### part a ###################### 
        D_coords = calib_data[frame]['vec_d']
        
        # frame for d -> D
        F_D = Calibration_Registration.point_cloud_reg( d_coords, D_coords )
        # homogeneous representation  
        F_D = transforms3d_extend.affines.compose( F_D['Trans'],
                                                  F_D['Rotation'], zoom )   
#         print(F_D)
        ###################### part b ###################### 
        A_coords = calib_data[frame]['vec_a']
        
        # frame a -> A
        F_A = Calibration_Registration.point_cloud_reg( a_coords, A_coords )     
        # homogeneous representation     
        F_A = transforms3d_extend.affines.compose( F_A['Trans'],
                                                  F_A['Rotation'], zoom )
#         print(F_A)  
        
        ###################### part c ###################### 
        F_C = transforms3d_extend.inverse_transform44( F_D ).dot( F_A ) 
        
        # compute C_expected in homogeneous then down-convert to 3
        C_expected = [F_C.dot( c )[:3] for c in c_coords]
        
        C_expected_frames.append( C_expected )

    # for

    ###################### part d ###################### 
    # write the output file for C_expected
    with open( outfile, 'w+' ) as writestream:
        outname = outfile.split( '/' )[-1]  # remove the path part
        writestream.write( "{0}, {1}, {2}\n".format( len( calbody['vec_c']),
                                                   len( frames ),
                                                   outname ) )  # first line
        
        writestream.write( "0, 0, 0\n0, 0, 0\n" )  # write place-holders for 
                                                # post position
        # write the C_expected values 
        for C_expected in C_expected_frames:
            for c in C_expected:
                writestream.write( "{0:.2f}, {1:.2f}, {2:.2f}\n".format( *c ) )
                
            # for
        # for
    # with
    print( "File '{}' saved.".format( outfile ) )
    
    return [C_expected_frames, outfile]

#    


# Calculate the position of dimple for empivot (problem 5)
def compute_DimplePos( filename_empivot : str ):
    """ This file
    """
    # attain the metadata from filename
    name_pattern = r'pa(.)-(debug|unknown)-(.)-empivot.txt'
    res_pattern = re.search( name_pattern, filename_empivot )
    assign_num, data_type, letter = res_pattern.groups()
    outfile = "../pa{0}_results/pa{0}-{1}-{2}-output{0}.txt".format( assign_num,
                                                                    data_type,
                                                                    letter )
    
    # open empivot file
    empivot = open_files.open_empivot( filename_empivot )
    frames = empivot.keys()
    N_frames = len( frames )
    
    ################## a ################
    # use first frame of pivot calibration data to define a local "probe" coordinate system
    G_first = empivot['frame1']
    G_zero = np.sum( G_first, axis = 0 ) / float( N_frames )
    g_j = G_first - G_zero
    
    Trans_empivot = []
    zoom = np.ones( 3 )  # for frame composition
    for frame in frames:
        ################## b ################
        # for each frame, compute transformation of F_G[k]
        G = empivot[frame]
        # frame transformation [g_j -> G]
        F_G = Calibration_Registration.point_cloud_reg( g_j, G )
        # homogenous representation
        F_G = transforms3d_extend.affines.compose( F_G['Trans'],
                                                  F_G['Rotation'], zoom )
        Trans_empivot.append( F_G )

    # pivot calibration
    t_G, p_post = Calibration_Registration.pointer_calibration( Trans_empivot )
    
    Dimple_positions = 1
    return Dimple_positions  # return p_post here?

# comptue_Dimplepos


# calculate the dimple position given optpivot (Problem 6) 
def perform_optical_pivot( optpivot_file: str ):
    """ This function reads in the optical pivot data file to perform
        the pivot calibration on the optical tracked pointer.
       
        @param calbody_file:    string representing the file name 
                                of the corresponding calibration body
                                data file.
                               
        @param optpivot_file:   string representing the file name 
                                of the corresponding optical pivot
                                data file.
                               
        @return: the 3-D vector of the optical pointer's position, t_h.
        
        @return: the 3-D vector of the optical pivot calibration point 
                 position.
    
    """
    # attain the metadata from filename
    name_pattern = r'pa(.)-(debug|unknown)-(.)-optpivot.txt'
    res_pattern = re.search( name_pattern, filename_optpivot )
    assign_num, data_type, letter = res_pattern.groups()
    #===========================================================================
    # outfile = "../pa{0}_results/pa{0}-{1}-{2}-output{0}.txt".format(assign_num,
    #                                                                 data_type,
    #                                                                 letter)
    #===========================================================================
    
    # open empivot file
    optpivot, em_data = open_files.open_optpivot( filename_optpivot )
    calbody = open_files.open_calbody( filename_calbody )
    frames = optpivot.keys()
    N_frames = len( frames )
    
    # get the d coordinates
    Hzero = np.mean( H['frame1'], axis = 0 )
    d_coords = calbody['vec_d']
    transform_list = []
    for frame in frames:
        ################## get F_D ##################  
        D_coords = em_data[frame]
        F_D = Calibration_Registration.point_cloud_reg( d_coords, D_coords )
        F_D = transforms3d_extend.affines.compose(F_D['Trans'], F_D['Rotation'],
                                                  np.ones(3))
        invF_D = transforms3d_extend.inverse_transform44(F_D)
        
        ################## compute invF_D.H ################## 
        # homogeneous rep.
        H_coords = [np.append( H_j, 1 ) for H_j in optpivot[frame]]
        d_H_coords = np.array( [invF_D.dot( H_j )[:3] for j_j in H_coords] )
        
        h_coords = [np.append( H_j - Hzero, 1 ) for H_j in optpivot[frame]]
        d_h_coords = np.array( [invF_D.dot( h_j )[:3] for j_j in h_coords] )
        
        ######### compute frame transform for d_h_i -> d_H_i ######### 
        ################### F_D H_j = F_k F_D h_j ####################
        transform_frame = Calibration_Registration.point_cloud_reg(d_h_coords, d_H_coords)
        transform44_frame = transforms3d_extend.affines.compose(transform_frame['Trans'],
                                                                transform_frame['Rotation'],
                                                                np.ones(3))
        # add 4x4 transformation to list for pivot calibration
        transform_list.append(transform44_frame)
         
    # for 
    
    # perform pivot calibration
    t_h, t_post = Calibration_Registration.pointer_calibration(transform_list)
    
    return t_h, t_post

# perform_optical pivot


def combine_postdata_tofile( filename: str, em_post: np.ndarray, opt_post: np.ndarray ):
    """This function is to combine the post data from the already written 
       output file from 'compute_Cexpected' for the optical and EM tracking
       systems. This file will follow the format as described in the 
       assignment.
       
        @author: Dimitri Lezcano
       
        @param filename:    string representing the filename to be written,
                            generated by 'compute_Cexpected'
                           
        @param em_post:     numpy column vector representing the post
                            position measured in the EMtracking pivot
                            calibration.
                           
        @param opt_post:    numpy column vector representing the post
                            position measured in the optical tracking
                            pivot calibration.
    """
    lines = []
    with open( filename, 'r' ) as readstream:
        for num, line in enumerate( readstream ):
            if num == 1:  # line #2 for EM_post 
                lines.append( "{0:.2f}, {1:.2f}, {2:.2f}\n".format( *em_post ) )
            
            elif num == 2:  # line #3 for OPT_post
                lines.append( "{0:.2f}, {1:.2f}, {2:.2f}\n".format( *opt_post ) )
                
            else:  # all other lines are fine
                lines.append( line )
            
    # with
    
    # write the altered output
    with open( filename, 'w+' ) as writestream:
        for line in lines:
            writestream.write( line )
            
    # with
    print( "File '{}' saved.".format( filename ) )
    
# combine_postdata_tofile


if __name__ == '__main__':
    calbody = "../pa1-2_data/pa1-debug-a-calbody.txt"
    calreadings = "../pa1-2_data/pa1-debug-a-calreadings.txt"
    empivot = "../pa1-2_data/pa1-debug-a-empivot.txt"
    
    compute_DimplePos( empivot )
    
    print( 20 * '=', "FUNCTIONALITY TEST: WRITE FILE", 20 * '=' )
    _, filename = compute_Cexpected( calbody, calreadings )
    em_test = np.array( [1.362, 1.457, 1.632] )
    opt_test = np.array( [2.456, 2.678, 2.891] )
    
    combine_postdata_tofile( filename, em_test, opt_test )
    
    print( 'Completed' )
