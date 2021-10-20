from os import listdir
from os.path import isdir, join
import os

def get_dir(path_loc, only_dir=True):
    result = []
    try:
        for f in listdir(path_loc):
            if f.startswith('.') or f.startswith('__'): #skip hidden folders
                continue
            if only_dir:
                if isdir(join(path_loc, f)):
                    result.append(f)
            else:
                result.append(f)
    except FileNotFoundError as fe:
        print('Error: ',fe)
    return result
    
def print_dir(path_loc, only_dir=True):
    for f in listdir(path_loc):
        if f.startswith('.') or f.startswith('__'): #skip hidden folders
            continue
        if only_dir:
            if isdir(join(path_loc, f)):
                print(join(path_loc, f))
        else:
            print(join(path_loc, f))
            
def get_file_size(filename):
    return os.stat(filename).st_size//1024