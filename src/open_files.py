""" Open data files
    The output format is dictionary
"""
import sys
import numpy as np

DEFAULT_DIR = "../pa1-2_data/"

def open_calbody(file_name):
    tmp = np.empty([1,3])
    with open(file_name,"r") as filestream: 
        for line in filestream:
            currentline = line.split(",")
            tmpArray = [float(currentline[0]), float(currentline[1]), float(currentline[2])]
            tmp = np.vstack((tmp, tmpArray))
    
    tmp = np.delete(tmp, 0,0)

    N_D = int(tmp[0,0])
    N_A = int(tmp[0,1])
    N_C = int(tmp[0,2])
    calbody = {'vec_d' : tmp[1 : 1+N_D, :],
               'vec_a' : tmp[1+N_D : 1+N_D+N_A, :],
               'vec_c' : tmp[1+N_D+N_A : 1+N_D+N_A+N_C, :]}

    return calbody

# open_calbody

def open_calreadings(file_name): 
    tmp = np.empty([1,3])
    with open(file_name,"r") as filestream: 
        lines = filestream.read().split('\n')
        N_D, N_A, N_C, N_frame, file_name = lines[0].split(',')
        for lIdx in range(1,len(lines)-1):
            currentline = lines[lIdx].split(",")
            tmpArray = [float(currentline[0]), float(currentline[1]), float(currentline[2])]
            tmp = np.vstack((tmp, tmpArray))
    
    tmp = np.delete(tmp, 0,0)
    N_D = int(N_D)
    N_A = int(N_A)
    N_C = int(N_C)
    files_p_frame = N_D + N_A + N_C
    N_frame = int(N_frame)

    calreadings = {}
    for fIdx in range(N_frame):
        start_idx = files_p_frame * fIdx 
        tmpDict = {'vec_d' : tmp[start_idx : start_idx + N_D, :],
                   'vec_a' : tmp[start_idx + N_D : start_idx + N_D + N_A, :],
                   'vec_c' : tmp[start_idx + N_D + N_A : start_idx + N_D + N_A + N_C, :]}
        calreadings['frame'+str(fIdx+1)] = tmpDict

    return calreadings
#########################
def open_calbody_npfloat(file_name):
    tmp = np.empty([1,3])
    with open(file_name,"r") as filestream: 
        for line in filestream:
            currentline = line.split(",")
            tmpArray = np.float32([currentline[0], currentline[1], currentline[2]])
            tmp = np.vstack((tmp, tmpArray))
    
    tmp = np.delete(tmp, 0,0)

    N_D = int(tmp[0,0])
    N_A = int(tmp[0,1])
    N_C = int(tmp[0,2])
    calbody = {'vec_d' : tmp[1 : 1+N_D, :],
               'vec_a' : tmp[1+N_D : 1+N_D+N_A, :],
               'vec_c' : tmp[1+N_D+N_A : 1+N_D+N_A+N_C, :]}

    return calbody

# open_calbody

def open_calreadings_npfloat(file_name): 
    tmp = np.empty([1,3])
    with open(file_name,"r") as filestream: 
        lines = filestream.read().split('\n')
        N_D, N_A, N_C, N_frame, file_name = lines[0].split(',')
        for lIdx in range(1,len(lines)-1):
            currentline = lines[lIdx].split(",")
            tmpArray = np.float32([currentline[0], currentline[1], currentline[2]])
            tmp = np.vstack((tmp, tmpArray))
    
    tmp = np.delete(tmp, 0,0)
    N_D = int(N_D)
    N_A = int(N_A)
    N_C = int(N_C)
    files_p_frame = N_D + N_A + N_C
    N_frame = int(N_frame)

    calreadings = {}
    for fIdx in range(N_frame):
        start_idx = files_p_frame * fIdx 
        tmpDict = {'vec_d' : tmp[start_idx : start_idx + N_D, :],
                   'vec_a' : tmp[start_idx + N_D : start_idx + N_D + N_A, :],
                   'vec_c' : tmp[start_idx + N_D + N_A : start_idx + N_D + N_A + N_C, :]}
        calreadings['frame'+str(fIdx+1)] = tmpDict

    return calreadings

#########################
# open_calreadings


def open_empivot(filename: str):
    """ A function that reads in the the data from an empivot.txt data file and returns
        a dictionary of the frames from the EM markers.
        
        @author: Dimitri Lezcano
        
        @param filename: string of the 'empivot' filename to be read
        
        @return: dictionary of each frame labelled 'frame1', 'frame2', ...
                 Each frame is a list of the coordinates for each EM marker.
        
    """ 
    
    with open(filename,"r") as file:
        lines = file.read().split('\n')
        
        # read in the first line
        N_EMmarks, N_frames, _ = lines[0].split(',')
        N_EMmarks = int(N_EMmarks) # int conversion
        N_frames = int(N_frames)   # int conversion
        
        # process thre frames
        empivot = {}
        for i in range(1, N_frames + 1):
            marker_coordinates = []
            for j in range(N_EMmarks):
                coords = np.fromstring(lines[i + j], dtype = float ,sep=',')
                marker_coordinates.append(coords) # add the position from the jth EM marker
            
            # for
                
            empivot['frame' + str(i)] = marker_coordinates 
        
        # for
        
    # with 
    return empivot
            
# open_empivot

def open_optpivot(filename: str):
    """ A function that reads in the the data from an optpivot.txt data file and returns
        two dictionaries of  the frames fo each of the optical and EM trackers.
        
        @author: Dimitri Lezcano
        
        @param filename: string of the 'optpivot' filename to be read
        
        @return: 2 dictionaries where each frame is labelled 'frame1', 'frame2', ...
                 Each frame is a list of the coordinates of each marker
                 for the EM and OPT markers
        
    """ 
    
    with open(filename,"r") as file:
        lines = file.read().split('\n')
        
        # read in the first line
        N_EMmarks, N_OPTmarks, N_frames, _ = lines[0].split(',')
        N_EMmarks = int(N_EMmarks)      # int conversion
        N_OPTmarks = int(N_OPTmarks)    # int conversion
        N_frames = int(N_frames)        # int conversion
        
        # process the frames
        empivot = {}
        optpivot = {}
        for i in range(1, N_frames + 1):
            #process EM coordinate frame
            emmarker_coordinates = []
            for j in range(N_EMmarks):
                coords = np.fromstring(lines[i + j], dtype = float ,sep=',')
                emmarker_coordinates.append(coords) # add the position from the jth EM marker
            
            # for
            empivot['frame' + str(i)] = emmarker_coordinates 
            
            #process OPT coordinate frame
            optmarker_coordinates = []
            for k in range(N_OPTmarks):
                coords = np.fromstring(lines[i + N_EMmarks + k], dtype = float ,sep=',')
                optmarker_coordinates.append(coords) # add the position from the jth OPT marker
                
            #for
            optpivot['frame' + str(i)] = optmarker_coordinates     
        
        # for
        
    # with 
    
    return [optpivot, empivot]
            
# open_optpivot
