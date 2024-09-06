import torch
import torch.nn as nn
from os.path import dirname, join as pjoin
from scipy.io import savemat, loadmat
import numpy as np


batch_size = 10

data_dir = './test_vws/MAT/SNR_-10/virtualwave_1.mat'
mat_file = loadmat(data_dir)

print(mat_file['admm_virt_space'].shape)
print(mat_file['fistanet_virt_space'].shape)



