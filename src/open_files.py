""" Open data files
    The output format is dictionary
"""
# import sys # unused
import numpy as np

DEFAULT_DIR = "../pa1-2_data/"


def open_calbody( file_name ):
    """ A function that reads in the the data from an calbody.txt data file and returns
        a dictionary of the positions of the markers on the calibration object.
        
        @author: Hyunwoo Song
        
        @param filename: string of the 'calbody' filename to be read
        
        @return: dictionary of each frame labelled 'vec_d', 'vec_a', 'vec_c'
                 Each refers to the coordinates of d_i, a_i, c_i.
        
    """ 
    tmp = np.empty( [1, 3] )
    with open( file_name, "r" ) as filestream: 
        for line in filestream:
            currentline = line.split( "," )
            tmpArray = [float( currentline[0] ), float( currentline[1] ), float( currentline[2] )]
            tmp = np.vstack( ( tmp, tmpArray ) )
    
    tmp = np.delete( tmp, 0, 0 )

    N_D = int( tmp[0, 0] )
    N_A = int( tmp[0, 1] )
    N_C = int( tmp[0, 2] )
    calbody = {'vec_d' : tmp[1 : 1 + N_D, :],
               'vec_a' : tmp[1 + N_D : 1 + N_D + N_A, :],
               'vec_c' : tmp[1 + N_D + N_A : 1 + N_D + N_A + N_C, :]}

    return calbody

# open_calbody


def open_calreadings( file_name ): 
    """ A function that reads in the the data from an calreadings.txt data file and returns
        a dictionary of the frames from the markers' position.
        
        @author: Hyunwoo Song
        
        @param filename: string of the 'calreadings' filename to be read
        
        @return: dictionary of each frame labelled 'frame1', 'frame2', ...
                 Each frame is a list of the coordinates for D_i, A_i, C_i.
        
    """ 
    tmp = np.empty( [1, 3] )
    with open( file_name, "r" ) as filestream: 
        lines = filestream.read().split( '\n' )
        N_D, N_A, N_C, N_frame, file_name = lines[0].split( ',' )
        for lIdx in range( 1, len( lines ) - 1 ):
            currentline = lines[lIdx].split( "," )
            tmpArray = [float( currentline[0] ), float( currentline[1] ), float( currentline[2] )]
            tmp = np.vstack( ( tmp, tmpArray ) )
    
    tmp = np.delete( tmp, 0, 0 )
    N_D = int( N_D )
    N_A = int( N_A )
    N_C = int( N_C )
    files_p_frame = N_D + N_A + N_C
    N_frame = int( N_frame )

    calreadings = {}
    for fIdx in range( N_frame ):
        start_idx = files_p_frame * fIdx 
        tmpDict = {'vec_d' : tmp[start_idx : start_idx + N_D, :],
                   'vec_a' : tmp[start_idx + N_D : start_idx + N_D + N_A, :],
                   'vec_c' : tmp[start_idx + N_D + N_A : start_idx + N_D + N_A + N_C, :]}
        calreadings['frame' + str( fIdx + 1 )] = tmpDict

    return calreadings

# open_calreadings


#########################
def open_calbody_npfloat( file_name ):
    """ This function is used to read the data as numpy float 64.
        The function of this is same as open_calbody
        
        @author: Hyunwoo Song
        
        @param filename: string of the 'calbody' filename to be read
        
        @return: dictionary of each frame labelled 'vec_d', 'vec_a', 'vec_c'
                 Each refers to the coordinates of d_i, a_i, c_i.
        
    """
    tmp = np.empty( [1, 3] )
    with open( file_name, "r" ) as filestream: 
        for line in filestream:
            currentline = line.split( "," )
            tmpArray = np.float64( [currentline[0], currentline[1], currentline[2]] )
            tmp = np.vstack( ( tmp, tmpArray ) )
    
    tmp = np.delete( tmp, 0, 0 )

    N_D = int( tmp[0, 0] )
    N_A = int( tmp[0, 1] )
    N_C = int( tmp[0, 2] )
    calbody = {'vec_d' : tmp[1 : 1 + N_D, :],
               'vec_a' : tmp[1 + N_D : 1 + N_D + N_A, :],
               'vec_c' : tmp[1 + N_D + N_A : 1 + N_D + N_A + N_C, :]}

    return calbody

# open_calbody_npfloat


def open_calreadings_npfloat( file_name ):  
    """ This function is used to read the data in numpy float64 format. 
        The function of this is same as open_calreadings        

        @author: Hyunwoo Song
        
        @param filename: string of the 'calreadings' filename to be read
        
        @return: dictionary of each frame labelled 'frame1', 'frame2', ...
                 Each frame is a list of the coordinates for D_i, A_i, C_i.
        
    """ 
    tmp = np.empty( [1, 3] )
    with open( file_name, "r" ) as filestream: 
        lines = filestream.read().split( '\n' )
        N_D, N_A, N_C, N_frame, file_name = lines[0].split( ',' )
        for lIdx in range( 1, len( lines ) - 1 ):
            currentline = lines[lIdx].split( "," )
            tmpArray = np.float64( [currentline[0], currentline[1], currentline[2]] )
            tmp = np.vstack( ( tmp, tmpArray ) )
    
    tmp = np.delete( tmp, 0, 0 )
    N_D = int( N_D )
    N_A = int( N_A )
    N_C = int( N_C )
    files_p_frame = N_D + N_A + N_C
    N_frame = int( N_frame )

    calreadings = {}
    for fIdx in range( N_frame ):
        start_idx = files_p_frame * fIdx 
        tmpDict = {'vec_d' : tmp[start_idx : start_idx + N_D, :],
                   'vec_a' : tmp[start_idx + N_D : start_idx + N_D + N_A, :],
                   'vec_c' : tmp[start_idx + N_D + N_A : start_idx + N_D + N_A + N_C, :]}
        calreadings['frame' + str( fIdx + 1 )] = tmpDict

    return calreadings

#########################
# open_calreadings_npfloat


def open_empivot( filename: str ):
    """ A function that reads in the the data from an empivot.txt data file and returns
        a dictionary of the frames from the EM markers.
        
        @author: Dimitri Lezcano
        
        @param filename: string of the 'empivot' filename to be read
        
        @return: dictionary of each frame labelled 'frame1', 'frame2', ...
                 Each frame is a list of the coordinates for each EM marker.
        
    """ 
    
    with open( filename, "r" ) as file:
        lines = file.read().split( '\n' )
        
        # read in the first line
        N_EMmarks, N_frames, _ = lines[0].split( ',' )
        N_EMmarks = int( N_EMmarks )  # int conversion
        N_frames = int( N_frames )  # int conversion
        
        # process thre frames
        empivot = {}
        for i in range( N_frames ):
            marker_coordinates = []
            for j in range( 1, N_EMmarks + 1 ):
                coords = np.fromstring( lines[i * N_EMmarks + j], dtype = float ,
                                         sep = ',' )
                marker_coordinates.append( coords )  # add the position from the jth EM marker
            
            # for
                
            empivot['frame' + str( i + 1 )] = np.array( marker_coordinates ) 
        
        # for
        
    # with 
    return empivot
            
# open_empivot


def open_optpivot( filename: str ):
    """ A function that reads in the the data from an optpivot.txt data file and returns
        two dictionaries of  the frames of each of the optical and EM trackers.
        
        @author: Dimitri Lezcano
        
        @param filename: string of the 'optpivot' filename to be read
        
        @return: 2 dictionaries where each frame is labelled 'frame1', 'frame2', ...
                 Each frame is a list of the coordinates of each marker
                 for the markers on the EM tracking system and 
                 the OPT markers on the pointer. 
        
    """ 
    
    with open( filename, "r" ) as file:
        lines = file.read().split( '\n' )
        
        # read in the first line
        N_EMmarks, N_OPTmarks, N_frames, _ = lines[0].split( ',' )
        N_EMmarks = int( N_EMmarks )  # int conversion
        N_OPTmarks = int( N_OPTmarks )  # int conversion
        N_frames = int( N_frames )  # int conversion
        
        # process the frames
        em_data = {}
        optpivot = {}
        for i in range( N_frames ):
            # process EM coordinate frame
            emmarker_coordinates = []
            for j in range( 1, N_EMmarks + 1 ):
                coords = np.fromstring( lines[i * ( N_OPTmarks + N_EMmarks ) + j],
                                         dtype = float , sep = ',' )
                emmarker_coordinates.append( coords )  # add the position from the jth EM marker
            
            # for
            em_data['frame' + str( i + 1 )] = np.array( emmarker_coordinates ) 
            
            # process OPT coordinate frame
            optmarker_coordinates = []
            for k in range( 1, N_OPTmarks + 1 ):
                coords = np.fromstring( lines[i * ( N_OPTmarks + N_EMmarks ) + N_EMmarks + k],
                                         dtype = float , sep = ',' )
                optmarker_coordinates.append( coords )  # add the position from the jth OPT marker
                
            # for
            optpivot['frame' + str( i )] = np.array( optmarker_coordinates )     
        
        # for
        
    # with 
    
    return [optpivot, em_data]
            
# open_optpivot


def open_ctfiducials( filename: str ):
    """ A function that reads in the the data from an ct-fiducials.txt data file and returns
        a numpy array of the fiducial coordinates in the CT frame of reference.
        
        @author: Dimitri Lezcano
        
        @param filename: string of the 'ct-fiducials' filename to be read
        
        @return: a numpy array of the ct-fiducial coordinates given in the filename
                 where each row is the coordinate of one of the fiducials
        
    """ 
    with open( filename, 'r' ) as file:
        lines = file.read().split( '\n' )
        N_B, _ = lines[0].split( ',' )
        N_B = int( N_B )
        
        B_coords = []
        for i in range( 1, N_B + 1 ):
            B = np.fromstring( lines[i], dtype = float , sep = ',' )
            B_coords.append( B )
            
        # for
        
    # with 
    
    B_coords = np.array( B_coords )  
    
    return B_coords

# open_ctfiducials


def open_emfiducials( filename: str ):
    """ A function that reads in the the data from an em-fiducialss.txt data file and returns
        a dictionary of frames with em fiducial coordinates.
        
        @author: Dimitri Lezcano
        
        @param filename: string of the 'em-fiducialss' filename to be read
        
        @return: a dictionary of frames with each frame consisting of a numpy 
                 array of the em-fiducial coordinates given in the filename
                 where each row is the coordinate of one of the fiducials.        
    """ 
    with open( filename, 'r' ) as file:
        lines = file.read().split( '\n' )
        N_G, N_frames, _ = lines[0].split( ',' )
        
        G_coords = {}
        for i in range( N_frames ):
            coords = []
            for j in range( 1, N_G + 1 ):
                G = np.fromstring( lines[i * N_G + j], dtype = 'float' ,
                                   sep = ',' )
                coords.append( G )
            # for    
            G_coords['frame' + str( i + 1 )] = np.array( coords )
            
        # for
        
        return G_coords
    
# open_emfiducials


def open_emnav( filename: str ):
    """ A function that reads in the the data from an em-nav.txt data file and returns
        a dictionary of  the frames of each of EM trackers.
        
        @author: Dimitri Lezcano
        
        @param filename: string of the 'em-nav' filename to be read
        
        @return: a dictionary of frames with each frame consisting of a numpy 
                 array of the em-navigation coordinates given in the filename
    """
    
    with open( filename, 'r' ) as file:
        lines = file.read().split( '\n' )
        N_G, N_frames, _ = lines[0].split( ',' )
        
        G_coords = {}
        for i in range( N_frames ):
            coords = []
            for j in range( 1, N_G + 1 ):
                G = np.fromstring( lines[i * N_G + j], dtype = 'float' ,
                                   sep = ',' )
                coords.append( G )
            # for    
            G_coords['frame' + str( i + 1 )] = np.array( coords )
            
        # for
        
        return G_coords

# open_emnav


def open_output1( filename: str ):
    """ A function that reads in the the data from an output1.txt data file and 
        returns a dictionary with 3 values, the optical probe position,
        the EM probe position, and a dictionary of the frames for C_expectee
        
        @author: Dimitri Lezcano
        
        @param filename: string of the 'output1' filename to be read
        
        @return: a dictionary with three keys:
                 "opt_probe":  the optical probe position
                 "em_probe":   the EM probe position
                 "C_expected": a dictionary of frames for the C_expected coordinates.
    """
    retval = {}
    with open( filename, 'r' ) as file:
        lines = file.read().split( '\n' )
        N_C, N_frames, _ = lines[0].split( ',' )
        N_C = int( N_C )
        N_frames = int( N_frames )
        
        em_probe_pos = np.fromstring( lines[1], dtype = 'float' ,
                                   sep = ',' )
        opt_probe_pos = np.fromstring( lines[2], dtype = 'float' ,
                                   sep = ',' )
        retval['em_probe'] = em_probe_pos
        retval['opt_probe'] = opt_probe_pos
        
        C_coords = {}
        for i in range( N_frames ):
            coords = []
            for j in range( 2, N_C + 2 ):
                c = em_probe_pos = np.fromstring( lines[i * N_C + j],
                                                   dtype = 'float' ,
                                                   sep = ',' )
                coords.append( c )
            
            # for 
            C_coords['frame' + str( i + 1)] = np.array(coords)
            
        # for
        retval['C_expected'] = C_coords
        
        return retval
    
# open_output1
