""" Open data files
    The output format is dictionary
    """
import sys
import numpy as np

DEFAULT_DIR = "/Users/songhyunwoo/Documents/JohnsHopkins/2019 Fall_Local/Computer Integrated Surgery/Programming1_2/PA 1-2 Student Data/"

def open_calbody(file_name):
    tmp = np.empty([1,3])
    with open(DEFAULT_DIR + file_name + "calbody.txt","r") as filestream:
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

def open_calreadings(file_name):
    tmp = np.empty([1,3])
    with open(DEFAULT_DIR + file_name + "calreadings.txt","r") as filestream:
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
