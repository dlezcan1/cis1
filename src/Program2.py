'''
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
from Program1 import compute_Cexpected, write_data, perform_optical_pivot


def improved_empivot_calib( filename_empivot: str , debug: bool = True ):
    """This function is to provide an imporved pivot calibration of the EM
       probe using the undistort function provided in Calibration_Registration
       to remove distortion in the EM field.
       
       Here the EM field will be undistorted using the C_expected values
       and then the EM pivot calibration will be performed. using the
       undistorted values.
       
       @author: Dimitri Lezcano
              
       @param filename_empivot:  a string representing the data file for the 
                                 EM pivot tracking to be read in.
                                 
       @param debug:             A boolean representing if should use debug
                                 output1 files or generated
                                 
       @return: t_g, t_post
                t_G:    the pointer's tip location
                t_post: the post position the pointer pivoted on.
       
    """
    # locate associated calreadings and output file for this file.
    file_pattern = r'pa2-(debug|unknown)-(.)-empivot.txt'
    file_fmt = '../pa1-2_data/pa2-{0}-{1}-{2}.txt'
    res_empivot = re.search( file_pattern, filename_empivot )
    data_type, letter = res_empivot.groups()
    filename_calreadings = file_fmt.format( data_type, letter, 'calreadings' )
    if debug:
        filename_output1 = file_fmt.format( data_type, letter, 'output1' )
        
    else:
        filename_output1 = '../pa2_results/pa2-{0}-{1}-{2}.txt'.format( data_type,
                                                                        letter,
                                                                         'output1' )
    # else
    
    # find the distortion coefficients
    coeffs, qmin, qmax = undistort_emfield( filename_calreadings, filename_output1, 5 )
    
#     print("[Before] coeffs, qmin, qmax\n", coeffs, qmin, qmax)
    # read in EM pivot data
    empivot = open_files.open_empivot( filename_empivot )
    
    # check if the box values are large enough, if not fix it.
    min_emdata = np.min( [frame for frame in empivot.values()] )
    max_emdata = np.max( [frame for frame in empivot.values()] )
    if min_emdata < qmin or max_emdata > qmax:
            print( "EMpivot_calib: qmin or qmax rule violated. Recalculating with larger box." )
            qmin = min( min_emdata, qmin )
            qmax = max( max_emdata, qmax )
            coeffs, qmin, qmax = undistort_emfield( filename_calreadings,
                                                     filename_output1, 5,
                                                     qmin, qmax )
            print( 'Recalibrated' )
            
#     print("[After] coeffs, qmin, qmax\n", coeffs, qmin, qmax)
    # if
    
    # correct empivot data
    empivot_calibrated = {}
    for frame in empivot.keys():
        coords = empivot[frame]
        coords_calib = [cr.correctDistortion( coeffs, v, qmin, qmax ) for v in coords]
        empivot_calibrated[frame] = np.array( coords_calib )
        
    # for
    
    # perform the EM pivot calibration with the calibrated data
    G_first = empivot_calibrated['frame1']
    G_zero = np.mean( G_first, axis = 0 )
    g_j = G_first - G_zero
    
    Trans_empivot = []
    zoom = np.ones( 3 )  # for frame composition
    for frame in empivot_calibrated.keys():
        ################## b ################
        # for each frame, compute transformation of F_G[k]
        G = empivot_calibrated[frame]
        # frame transformation [g_j -> G]
        F_G = cr.point_cloud_reg( g_j, G )
        # homogeneous representation
        F_G = tf3e.affines.compose( F_G['Trans'],
                                    F_G['Rotation'], zoom )
        Trans_empivot.append( F_G )

#     print("Trans_empivot \n", Trans_empivot)
    ############## c ################
    # pivot calibration
    t_G, p_post = cr.pointer_calibration( Trans_empivot )
    
    return t_G, p_post

# improved_empivot_calib


def undistort_emfield( filename_calreadings, filename_output1: str, order_fit: int ,
                       qmin = None, qmax = None ):
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
                                        
       @param qmin (optional):   a floating point number representing the min
                                 value for scaling
     
       @param qmax (optional):   a floating point number representing the min
                                 value for scaling

                                 
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
        
    coeffs, qmin, qmax = cr.undistort( C_read, C_expected, order_fit, qmin, qmax )
    
    return coeffs, qmin, qmax
    
# undistort_emfield

def correct_C(filename_calreadings : str, coef : np.ndarray, qmin, qmax):
    """ This functions corrects the C values from calreading txt file with respect to
        EM tracker base coordinate system.

        @author: Hyunwoo Song

        @param filename_em_fiducials: string of the filename to be read

        @param coef : coefficient for distortion correction

        @param qmin, qmax: A floating point number representing the min/max
                                 value for scaling 

        @return: position(x,y,z) of the fiducial points
    """

    name_pattern = r'pa(.)-(debug|unknown)-(.)-calreadings.txt'
    res_calreading = re.search(name_pattern, filename_calreadings)
    assign_num, data_type, letter = res_calreading.groups()
    outfile = "../pa{0}_results/pa{0}-{1}-{2}-output1.txt".format(assign_num,
                                                                    data_type,
                                                                    letter)

    Cal_readings = open_files.open_calreadings(filename_calreadings)
    C_undistorted = []
    for idx, frames in enumerate(Cal_readings):
#         print('frame %d/%d' %(idx+1, len(Cal_readings.keys())))
        C_distorted = Cal_readings[frames]['vec_c']

        #print("C_distorted \n", C_distorted)
        retval = np.array([cr.correctDistortion(coef, C_tmp, qmin, qmax) for C_tmp in C_distorted])
       
        #print("C_corrected \n", np.array(retval))
        C_undistorted.append( retval )
        
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

def compute_Freg( filename_ctfiducials: str, filename_emfiducials: str , debug: bool = False ):
    """Function in order to compute the registration frame transformation
       from the fiducials data and the em-tracked pointer. 
    
       @author: Dimitri Lezcano
       
       @param filename_ctfiducials:  A string representing the ct-fiducials data
                                     file.
                                     
       @param filename_emfiducials:  A string representing the em-fiducials data
                                     file.
                                     
       @param debug:                 A boolean representing if should use debug
                                     output1 files or generated
                                     
       @return: F_reg, a 4x4 homogeneous transformation matrix corresponding
                to the transformation for the CT coordinates.
       
    """
    file_pattern = r'pa2-(debug|unknown)-(.)-ct-fiducials.txt'
    file_fmt = '../pa1-2_data/pa2-{0}-{1}-{2}.txt'
    res_empivot = re.search( file_pattern, filename_ctfiducials )
    data_type, letter = res_empivot.groups()
    filename_empivot = file_fmt.format( data_type, letter, 'empivot' )
    filename_calreadings = file_fmt.format( data_type, letter, 'calreadings' )
    if debug:
        filename_output1 = file_fmt.format( data_type, letter, 'output1' )
        
    else:
        filename_output1 = '../pa2_results/pa2-{0}-{1}-{2}.txt'.format( data_type,
                                                                        letter,
                                                                         'output1' )
    # else
    
    fid_ct = open_files.open_ctfiducials( filename_ctfiducials )
    fid_em = open_files.open_emfiducials( filename_emfiducials )
    
    # perform empivot calibration
    t_G, _ = improved_empivot_calib( filename_empivot , debug )
    t_G_hom = np.append( t_G, 1 )  # homogeneous representation
    
    coeffs, qmin, qmax = undistort_emfield( filename_calreadings, filename_output1, 5 )
    
    # check if qmin and qmax are not violated
    min_emdata = np.min( [frame for frame in fid_em.values()] )
    max_emdata = np.max( [frame for frame in fid_em.values()] )
    if min_emdata < qmin or max_emdata > qmax:
            qmin = min( min_emdata, qmin )
            qmax = max( max_emdata, qmax )
            coeffs, qmin, qmax = undistort_emfield( filename_calreadings,
                                                     filename_output1, 5,
                                                     qmin, qmax )
            print( "Freg: qmin or qmax rule violated. Recalculating with larger box." )
            
    # if
    
    # correct em_fiducial data
    fid_em_calibrated = {}
    for frame in fid_em.keys():
        coords = fid_em[frame]
        coords_calib = [cr.correctDistortion( coeffs, v, qmin, qmax ) for v in coords]
        fid_em_calibrated[frame] = np.array( coords_calib )
        
    # for
    
    # initialize stuff for pose determination of EM point
    G_first = fid_em_calibrated['frame1']
    G_zero = np.mean( G_first, axis = 0 )
    g_j = G_first - G_zero
    zoom = np.ones( 3 )  # no zooming
    
    B_matrix = np.zeros( ( 0, 3 ) )
    for frame in fid_em_calibrated.keys():
        # for each frame, caompute transformation of F_G[k]
        G = fid_em_calibrated[frame]
        # frame transformation [g_j -> G]
        F_G = cr.point_cloud_reg( g_j, G )
        # homogeneous representation
        F_G = tf3e.affines.compose( F_G['Trans'],
                                    F_G['Rotation'], zoom )
        
        B_i = F_G.dot( t_G_hom )[:3]
        B_matrix = np.vstack( ( B_matrix, B_i ) )
        
    # for
    
    Freg = cr.point_cloud_reg( B_matrix, fid_ct )
    Freg = tf3e.affines.compose( Freg['Trans'], Freg['Rotation'],
                                 zoom )
    return Freg

# compute_Freg

def compute_test_points(filename_emnav:str, coeffs, qmin, qmax, t_G, Freg):
    """
        This function computes the tip location with respect to the CT image

        @author: Hyunwoo Song

        @param filename_emnav: The name of input file where describes the frame
                                of data defining test points

        @param coeffs        : The coefficient which defines the distortion correction

        @param qmin, qmax    : A floating point number representing the min/max
                                 value for scaling
     
        @param t_G           : Vector representing the position of the pointer

        @param Freg          : Transformation between ct coordinate and em coordinate

        @return v            : The computed positions of test points in ct coordinates

    """
    name_pattern = r'pa2-(debug|unknown)-(.)-EM-nav.txt'
    res_emnav = re.search(name_pattern, filename_emnav)
    data_type, letter = res_emnav.groups()
    outfile = "../pa2_results/pa2-{0}-{1}-output2.txt".format(    data_type,
                                                                    letter)


    G_coords = open_files.open_emnav(filename_emnav)
    G_emnav_calib = {}
    for frame in G_coords.keys():
        G_tmp = G_coords[frame]
        G_calib = [cr.correctDistortion( coeffs, g, qmin, qmax) for g in G_tmp]
        G_emnav_calib[frame] = np.array(G_calib)

    G_first = G_emnav_calib['frame1']
    G_zero =np.mean( G_first, axis = 0 )
    g_j = G_first - G_zero
    zoom = np.ones(3)

    t_G_hom = np.append( t_G, 1 )  # homogeneous representation
    V_matrix = []
    for frame in G_emnav_calib.keys():
        G = G_emnav_calib[frame]

        F_G = cr.point_cloud_reg( g_j, G)
        #homogernous representation
        F_G = tf3e.affines.compose( F_G['Trans'],
                                    F_G['Rotation'], zoom )

        V_tmp = F_G.dot( t_G_hom)
        V_matrix.append(V_tmp)

    V_matrix = np.array(V_matrix)
    #compute test points
    v = np.array([Freg[:3].dot(v_tmp) for v_tmp in V_matrix])
    #v = Freg[:3].dot(V_matrix)
#     print("v (CT coordinate of pointer tip) \n", v)

    with open(outfile, 'w+') as writestream:
        outname = outfile.split('/')[-1]
        writestream.write("{0}, {1} \n".format(len(G_emnav_calib.keys()),
                                               outname) )

        #write the v values
        for v_calc in v:
            writestream.write( "{0:.2f}, {1:.2f}, {2:.2f}\n".format(*v_calc))

    print(" File '{}' written.".format(outfile))

    return v

# compute_test_points

if __name__ == '__main__':
#     # test compute_fiducial_pos
#     file_name_emfiducial = "../pa1-2_data/pa2-debug-a-em-fiducialss.txt"
#     file_name_calreadings = "../pa1-2_data/pa2-debug-a-calreadings.txt"
#     file_name_output1 = "../pa1-2_data/pa2-debug-a-output1.txt"
#     coef, qmin, qmax = undistort_emfield( file_name_calreadings, file_name_output1, 5 )
#     compute_test_points( file_name_emfiducial, coef, qmin, qmax )
    
    # main program
    calbody_list = sorted( glob.glob( "../pa1-2_data/*pa2*calbody.txt" ) )
    calreading_list = sorted( glob.glob( "../pa1-2_data/*pa2*calreadings.txt" ) )
    empivot_list = sorted( glob.glob( "../pa1-2_data/*pa2*empivot.txt" ) )
    optpivot_list = sorted( glob.glob( "../pa1-2_data/*pa2*optpivot.txt" ) )
    emfiducial_list = sorted( glob.glob( "../pa1-2_data/*pa2*em-fiducialss.txt" ) )
    emnav_list = sorted( glob.glob( "../pa1-2_data/*pa2*EM-nav.txt" ) )
    ctfid_list = sorted( glob.glob( "../pa1-2_data/*pa2*ct-fiducials.txt" ) )
    
    for calbody, calreadings, empivot, optpivot, emnav, ctfid, emfid in zip( calbody_list,
                                                                             calreading_list,
                                                                             empivot_list,
                                                                             optpivot_list,
                                                                             emnav_list,
                                                                             ctfid_list,
                                                                             emfiducial_list ):
        
        name_pattern = r'pa2-(debug|unknown)-(.)-calbody.txt'
        res_calbody = re.search( name_pattern, calbody )
        data_type, letter = res_calbody.groups()
        print( "Data set: {0}-{1}".format( data_type, letter ) )
        # compute C_expected and write output1
        C_expected, outfile = compute_Cexpected( calbody, calreadings )
        _, t_opt_post = perform_optical_pivot( calbody, optpivot )
        t_G, t_em_post = improved_empivot_calib( empivot, False )
        Freg = compute_Freg(ctfid, emfid, False)
        outfile1 = write_data( outfile, t_em_post, t_opt_post )
        coef, qmin, qmax = undistort_emfield( calreadings, outfile1, 5 )
        compute_test_points( emnav, coef, qmin, qmax , t_G, Freg)
        print( 'Completed\n' )
    pass
