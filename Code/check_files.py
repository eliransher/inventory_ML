import numpy as np
import os
import pickle as pkl


path = '/home/elirans/scratch/ph_samples'

files = os.listdir(path)

for fold in files:
    curr_path = os.path.join(path, fold)
    pikle_files = os.listdir(curr_path)
    
    for fil in pikle_files:
        full_path = os.path.join(curr_path, fil)
        try:
            with open(full_path, 'rb') as f:
                data = pkl.load(f)

        except Exception as e:
            print(f"Error loading file {full_path}: {e}")
            os.remove(full_path)