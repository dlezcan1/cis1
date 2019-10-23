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
                 position of the calibration body for each of the f
                 B
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
    
    C_expected_frames = []
    # start to iterate over the frames getting the required points
    for frame in frames: 
        ###################### part a ###################### 
        d_coords = calbody['vec_d']
        D_coords = calib_data[frame]['vec_d']
        
        # frame for d -> D
        F_D = Calibration_Registration.point_cloud_reg( d_coords, D_coords )
        # homogenous representation  
        F_D = transforms3d_extend.affines.compose( F_D['Trans'],
                                                  F_D['Rotation'], zoom )   
#         print(F_D)
        ###################### part b ###################### 
        a_coords = calbody['vec_a']
        A_coords = calib_data[frame]['vec_a']
        
        # frame a -> A
        F_A = Calibration_Registration.point_cloud_reg( a_coords, A_coords )     
        # homogenous representation     
        F_A = transforms3d_extend.affines.compose( F_A['Trans'],
                                                  F_A['Rotation'], zoom )
#         print(F_A)  
        
        ###################### part c ###################### 
        # convert c coordinates to homogenous vectors
        c_coords = [np.append( c, 1 ) for c in calbody['vec_c']]
        F_C = transforms3d_extend.inverse_transform44( F_D ).dot(F_A) 
        
        # compute C_expected in homogenous then down-convert to 3
        C_expected = [F_C.dot(c)[:3] for c in c_coords]
        
        C_expected_frames.append( C_expected )

    # for

    ###################### part d ###################### 
    # write the output file for C_expected
    with open( outfile, 'w+' ) as writestream:
        outname = outfile.split('/')[-1]        # remove the path part
        writestream.write("{0}, {1}, {2}\n".format(len(calib_data['frame1']),
                                                   len(frames),
                                                   outname)) # first line
        
        #writestream.write("0, 0, 0\n0, 0, 0\n") # write place-holders for 
                                                # post position
        # write the C_expected values 
        for C_expected in C_expected_frames:
            for c in C_expected:
                writestream.write("{0:.2f}, {1:.2f}, {2:.2f}\n".format(*c))
                
            #for
        #for
    # with
    print("File {}: saved.".format(outfile))
    
    return C_expected_frames, outfile
    

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
    name_pattern = r'pa(.)-(debug|unknown)-(.)-empivot.txt'
    res_calbody = re.search( name_pattern, filename_empivot )
    assign_num, data_type, letter = res_calbody.groups()
    
    # open empivot file
    empivot = open_files.open_empivot( filename_empivot )
    frames = empivot.keys()
    N_frames = len(frames)
    
    ################## a ################
    # use first frame of pivot calibration data to define a local "probe" coordinate system
    G_first = empivot['frame1']
    G_zero = np.mean(G_first, axis = 0)
    g_j = G_first - G_zero
    
    Trans_empivot = []
    zoom = np.ones(3) # for frame composition
    for frame in frames:
        ################## b ################
        # for each frame, compute transformation of F_G[k]
        G = empivot[frame]
        # frame transformation [g_j -> G]
        F_G = Calibration_Registration.point_cloud_reg(g_j, G)
        # homogenous representation
        F_G = transforms3d_extend.affines.compose( F_G['Trans'],
                                                  F_G['Rotation'], zoom)
        Trans_empivot.append(F_G)

    ############## c ################
    # pivot calibration
    t_G, p_post = Calibration_Registration.pointer_calibration(Trans_empivot)

    return p_post

def write_data(outfile, EM_probe_pos, OPT_probe_pos):
    """ This function writes the calculated data to output.txt
        From the .txt file written at the comput_Cexpected function,
        this function overwrites the EM_probe_pos and OPT_probe_pos
    """
    line_idx = 1
    lines = None
    insertline_em = " ".join(str(x) for x in EM_probe_pos)
    insertline_opt = " ".join(str(x) for x in OPT_probe_pos)
    insertline = "\n".join([insertline_em, insertline_opt]) + "\n"

    with open(outfile, 'r') as resultstream:
        outname = outfile.split('/')[-1]
        lines = resultstream.readlines()

    lines.insert(line_idx, insertline)

    with open(outfile, 'w+') as resultstream:
        resultstream.writelines(lines)
    
    return 0

if __name__ == '__main__':
    calbody = "../pa1-2_data/pa1-debug-a-calbody.txt"
    calreadings = "../pa1-2_data/pa1-debug-a-calreadings.txt"
    empivot = "../pa1-2_data/pa1-debug-a-empivot.txt"
    
    print("Compute C_expected")
    C_expected, outfile = compute_Cexpected(calbody, calreadings)
    print("Compute_EM probe position")
    EM_probe_pos = compute_DimplePos(empivot)
    print(EM_probe_pos)
    write_data(outfile, EM_probe_pos, [0, 0, 0])
    print('Completed')
