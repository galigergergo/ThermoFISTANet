from loader_ndt import DataSplit
import numpy as np
import torch
from M1LapReg import MatMask
from M5FISTANet import FISTANet
import argparse
from solver import Solver
from os.path import dirname, join as pjoin
import os
from scipy.io import loadmat


# =============================================================================
# Load dataset
# some parameters of the dataset
nr_diff_imgs = 10000
test_split = 0.05
val_split = 0.05

nr_diff_imgs = 20 / 20
test_split = 0
val_split = 0

fista_net_mode = 1      # 0, test mode; 1, train mode.

batch_size = 10   # 10
LayerNo = 4
FeatureNo = 16

num_epochs = 50
test_epoch = 31
start_epoch = 0

lr_dec_after = 10
lr_dec_every = 1

# load input and target dataset; test_loader, train_loader, val_loader
root_dir = "D:\\USERS\\galiger.gergo\\TrainData\\%d" % nr_diff_imgs
root_dir = "D:\\USERS\\galiger.gergo\\for_vw_benchmarking\\RotatedData_filt_2000\\Deg_45\\MAT"
#root_dir = 'C:\\Users\\galiger.gergo\\Desktop\\PlotData'
data_dir = "../data/NDTData/"
data_dir = "D:\\USERS\\galiger.gergo\\for_vw_benchmarking\\RotatedData_filt_2000\\Deg_45"

padding = 21
Nx = 64

# print('main run now...')

# =============================================================================
#
# Get 100 batch of test data and validate data
if __name__ == '__main__':
    train_loader, val_loader, test_loader, snr_nr = DataSplit(root_dir, nr_diff_imgs,
                                                              batch_size=batch_size,
                                                              validation_split=val_split,
                                                              test_split=test_split)
    test_data_1 = []
    test_data_2 = []

    # =============================================================================
    # Model 5
    # FISTA-Net: Proposed method

    print('===========================================')
    print('FISTA-Net...')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    #device = 'cuda'

    # load sensitivity matrix of EIT; load mask matrix, dim from 3228 to 4096
    # K matrix beolvasasa
    # CIKK: A_y matrix from (1), later just A (16) --\/ (K kernel matrix nálunk)
    # np.loadtxt(pjoin(data_dir, "Jmat.csv"), delimiter=",", dtype=float)
    #J_ndt = np.loadtxt(pjoin(data_dir, "kernel_abel_matrix.csv"), delimiter=",", dtype=float)
    fle = pjoin(data_dir, 'kernel_matrix.mat')
    mat_file = loadmat(fle)
    J_ndt = mat_file['K']
    #J_ndt = torch.from_numpy(J_ndt[:64,:])
    J_ndt = torch.from_numpy(J_ndt[:48,:])
    J_ndt = J_ndt.clone().detach().to(device=device)
    # J_ndt = torch.tensor(J_ndt, dtype=torch.float32, device=device)

    # row and column difference matrix for TV
    # TODO: [51] cikkből Laplacian difference matrix - egységmátrixra (később esetleg curvelet)
    # CIKK: L Laplacian matrix (16) --\/
    DMts = []  # np.eye(64)  # np.loadtxt(pjoin(data_dir, "DM.csv"), delimiter=",", dtype=float)
    # DMts = torch.from_numpy(DM)
    # DMts = DMts.clone().detach().to(device=device) #
    # DMts = torch.tensor(DMts, dtype=torch.float32, device=device)

    # CIKK: kör maszkolása a négyzet mátrixból --\/
    mask = MatMask(64)
    mask = torch.from_numpy(mask)
    mask = mask.clone().detach()  # torch.tensor(mask, dtype=torch.float32, device=device)

    fista_net = FISTANet(LayerNo, FeatureNo, J_ndt, DMts, mask)
    fista_net = fista_net.to(device)

    print('Total number of parameters fista net:',
          sum(p.numel() for p in fista_net.parameters()))

    # define arguments of fista_net
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='FISTANet')
    parser.add_argument('--num_epochs', type=int, default=num_epochs)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data_dir', type=str, default=data_dir)
    parser.add_argument('--save_path', type=str, default='./models_copy/FISTANet/')
    parser.add_argument('--start_epoch', type=int, default=start_epoch)
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--device', default=device)
    parser.add_argument('--log_interval', type=int, default=20)  # 20
    parser.add_argument('--test_epoch', type=int, default=test_epoch)

    parser.add_argument('--lr_dec_after', type=int, default=lr_dec_after)
    parser.add_argument('--lr_dec_every', type=int, default=lr_dec_every)
    # NDT arguments
    parser.add_argument('--Nx', type=int, default=Nx)
    parser.add_argument('--padding', type=int, default=padding)
    args = parser.parse_args()

    solver = Solver(fista_net, train_loader, val_loader, snr_nr, batch_size, args, (test_data_1, test_data_2))

    solver.load_model(test_epoch)
    avg_test_losses_per_snr_FISTANet, avg_test_losses_per_snr_ADMM = solver.test_MSE(train_loader, test_epoch)

    test_results_path = './test_losses'
    if not os.path.exists(test_results_path):
        os.makedirs(test_results_path)
    f = pjoin(test_results_path, 'MSE_per_SNR_FISTA-Net_ADMM_imgs_{}_epoch_{}.ckpt'.format(nr_diff_imgs, test_epoch))
    checkpoint = {
        'avg_test_losses_per_snr_ADMM': avg_test_losses_per_snr_ADMM,
        'avg_test_losses_per_snr_FISTANet': avg_test_losses_per_snr_FISTANet}
    torch.save(checkpoint, f)
