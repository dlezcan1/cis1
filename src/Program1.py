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
        
        #writestream.write("0, 0, 0\n0, 0, 0\n") # write place-holders for 
                                                # post position
        # write the C_expected values 
        for C_expected in C_expected_frames:
            for c in C_expected:
                writestream.write( "{0:.2f}, {1:.2f}, {2:.2f}\n".format( *c ) )
                
            # for
        # for
    # with
    print( "File '{}' written.".format( outfile ) )
    
    return [C_expected_frames, outfile]

#    

# Calculate the position of dimple
def compute_DimplePos(filename_empivot : str):
    """ This function returns the calibrated position of the dimple. 
        This function is for problem 5.
        
            The function use first frame of pivot data to define local 
        position and use this to compute g, relative position to the midpoint of
        markers. Then it uses point cloud function to calculate the transformation
        between g and the read data G.
            After getting the transformation at each frame, we calibrate the position
        of the position of the dimple using least-square method.

        @author: Hyunwoo Song    
    
        @param filename_empivot:    takes a string of the file name for 
                                    the EM markers reading 
        
        @return: p_post, the calibrated position of the dimple

    """
    # attain the metadata from filename
#     name_pattern = r'pa(.)-(debug|unknown)-(.)-empivot.txt'     # unused
#     res_calbody = re.search( name_pattern, filename_empivot )   # unused
#     assign_num, data_type, letter = res_calbody.groups()        # unused
    
    # open empivot file
    empivot = open_files.open_empivot( filename_empivot )
    frames = empivot.keys()
#     N_frames = len( frames )                                    # unuseds
    
    ################## a ################
    # use first frame of pivot calibration data to define a local "probe" coordinate system
    G_first = empivot['frame1']
    G_zero = np.mean(G_first, axis = 0)
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

    ############## c ################
    # pivot calibration
    t_G, p_post = Calibration_Registration.pointer_calibration( Trans_empivot )
    
    
    return t_G, p_post

# compute_Dimplepos


# calculate the dimple position given optpivot (Problem 6) 
def perform_optical_pivot( filename_calbody: str, filename_optpivot: str ):
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
    # open empivot file
    optpivot, em_data = open_files.open_optpivot( filename_optpivot )
    calbody = open_files.open_calbody( filename_calbody )
    frames = optpivot.keys()
    
    # get the d coordinates
    Hzero = np.mean( optpivot['frame1'], axis = 0 )
    h_coords = [np.append( H1_j - Hzero, 1 ) for H1_j in optpivot['frame1']]
    d_coords = calbody['vec_d']
    transform_list = []
    
    for frame in frames:
        ################## get F_D ##################  
        D_coords = em_data[frame]
        F_D = Calibration_Registration.point_cloud_reg( d_coords, D_coords )
        F_D = transforms3d_extend.affines.compose(F_D['Trans'], F_D['Rotation'],
                                                  np.ones(3))
        
        ################## compute invF_D.H ################## 
        invF_D = transforms3d_extend.inverse_transform44(F_D)
        
        # homogeneous rep.
        H_coords = [np.append( H_j, 1 ) for H_j in optpivot[frame]]
        d_H_coords = np.array( [invF_D.dot( H_j )[:3] for H_j in H_coords] )
        
        d_h_coords = np.array( [invF_D.dot( h_j )[:3] for h_j in h_coords] )
        
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

# will remove. This function is not used.
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



def write_data(outfile, EM_probe_pos, OPT_probe_pos):
    """ This function writes the calculated data to output.txt
        From the .txt file written at the comput_Cexpected function,
        this function overwrites the EM_probe_pos and OPT_probe_pos
    """
    line_idx = 1
    lines = None
    insertline_em = "{0:.2f}, {1: .2f}, {2: .2f}".format(*EM_probe_pos)
    insertline_opt = "{0: .2f}, {1: .2f}, {2: .2f}".format(*OPT_probe_pos)
    #insertline_em = ", ".join(str(x) for x in EM_probe_pos)
    #insertline_opt = ", ".join(str(x) for x in OPT_probe_pos)
    insertline = "\n".join([insertline_em, insertline_opt]) + "\n"

    with open(outfile, 'r') as resultstream:
        outname = outfile.split('/')[-1]
        lines = resultstream.readlines()

    lines.insert(line_idx, insertline)

    with open(outfile, 'w+') as resultstream:
        resultstream.writelines(lines)
    
    print("Result file saved >>> [ ", outname, "]")
    return 0

# write_data

if __name__ == '__main__':
    calbody_list = sorted(glob.glob("../pa1-2_data/*pa1*calbody.txt"))
    calreading_list = sorted(glob.glob("../pa1-2_data/*pa1*calreadings.txt"))
    empivot_list = sorted(glob.glob("../pa1-2_data/*pa1*empivot.txt"))
    optpivot_list = sorted(glob.glob("../pa1-2_data/*pa1*optpivot.txt"))
 
    
    
    for calbody, calreadings, empivot, optpivot in zip(calbody_list,
                                                       calreading_list,
                                                       empivot_list,
                                                       optpivot_list):
        
        name_pattern = r'pa1-(debug|unknown)-(.)-calbody.txt'
        res_calbody = re.search( name_pattern, calbody )
        _, letter = res_calbody.groups()
        print("Data set: ", letter)
        # compute C_expected (P4)
        C_expected, outfile = compute_Cexpected(calbody, calreadings)
        # compue em probe position (P5)
        _, EM_probe_pos = compute_DimplePos(empivot)
        # compute opt probe position (P6)
        _, OPT_probe_pos = perform_optical_pivot(calbody, optpivot)
        # write the result
        write_data(outfile, EM_probe_pos, OPT_probe_pos)

        print('Completed\n')
        
    # for
# if