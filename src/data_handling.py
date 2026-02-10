# -*- coding: utf-8 -*-
import os
import csv

##!!! CREATE A CLASS
# class FileHandler:
    # """ Class for handling files (e.g., saving and loading)"""
    # def __init__(self, filename, filetype, parent):
    #     self.parent = parent
    #     super(FileHandler, self).__init__()

def modify_filename(filename, a, b = '', ext = ''):
    """ 
    Modify filename:
        - add {a} to start of name
        - add {b} to end of name (if given)
        - add {ext} as extension
    """
    name, _ = os.path.splitext(filename)
    
    path_split=name.split('/')[0:-1] # leave only filepath (remove name)
    path='\\'.join(path_split) # re-join into string
    
    end_nameonly=name.split('/')[-1] # only filename
    if b != '':
        end = f"{a}_{end_nameonly}_{b}" # name with added info a and b
    else:
        end = f"{a}_{end_nameonly}" # name with added info a
    name_mod = f"{path}\\{end}{ext}"
    return name_mod

def _get_unique_filename(base_filename):
    """
    If 'filename.txt' exists, make 'filename_2.txt', etc.
    """
    ext='.txt'
    if not os.path.exists(f"{base_filename}{ext}"): ## search for filename.txt
        return base_filename

    i = 2
    while os.path.exists(f"{base_filename}_{i}"):
        i += 1
    return f"{base_filename}_{i}"
