import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys

sys.path.append('/home/eliransc/projects/def-dkrass/eliransc/mom_analysis/code')
from utils import *
from tqdm import tqdm
import shutil

import tarfile
from pathlib import Path

path_dump = '/home/elirans/scratch/inv/batch_S_17'




path = '/home/management/scratch/elirans/inv/a_S_16_lower'


files = os.listdir(path)

batch_size = 16
num_batches = int(len(files) / batch_size)

rand_num = np.random.randint(1, 1000000)

for batch_num in tqdm(range(num_batches)):

    input_ = np.array([])
    output_ = np.array([])
    for batch_ind in range(batch_size):
        file_num = batch_num * batch_size + batch_ind
        try:
            inp1, out1 = pkl.load(open(os.path.join(path, files[file_num]), 'rb'))
            s = int(files[file_num].split('_')[1])
            S = int(files[file_num].split('_')[2])
            inp = np.concatenate((np.log(inp1[0][:10]), np.log(inp1[1][:10]), np.array([s, S])))
            out = out1[1][:18]
        except:
            print('keep the same')
        if batch_ind > 0:
            input_ = np.concatenate((input_, inp.reshape(1, inp.shape[0])), axis=0)
            output_ = np.concatenate((output_, out.reshape(1, out.shape[0])), axis=0)
        else:
            input_ = inp.reshape(1, inp.shape[0])
            output_ = out.reshape(1, out.shape[0])

    batch_name =  'ariel_lead_no_neg_batch_17_lower_' + str(batch_num) + '.pkl'
    pkl.dump((input_, output_), open(os.path.join(path_dump, batch_name), 'wb'))

