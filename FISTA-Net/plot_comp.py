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
import matplotlib.pyplot as plt
import cv2


nr_diff_imgs = 10000

# load input and target dataset; test_loader, train_loader, val_loader
root_dir = "D:\\USERS\\galiger.gergo\\TrainData\\%d" % nr_diff_imgs
data_dir = "../data/NDTData/"

LayerNo = 4
FeatureNo = 16

padding = 21
Nx = 64

px = 1/plt.rcParams['figure.dpi']

def eval_model(model, sample_id, snr_id, epoch, root_dir):
    f = pjoin('./models_copy/FISTANet/', 'epoch_{}.ckpt'.format(epoch))
    checkpoint = torch.load(f)
    model.load_state_dict(checkpoint['model'])

    vwdir = pjoin(root_dir, 'VirtualWaves')
    snr_dir = os.listdir(vwdir)[snr_id]
    sample_dir = pjoin(vwdir, snr_dir)
    sample_dir = pjoin(sample_dir, 'MAT')

    fle = pjoin(sample_dir, 'virtualwave_%d.mat' % (sample_id + 1))
    mat_file = loadmat(fle)
    x_in = torch.from_numpy(mat_file['T_noisy'])
    x_img = torch.from_numpy(mat_file['T_virt_in_abelspace'])
    y_target = torch.from_numpy(mat_file['T_virtual_abel'])

    x_in = torch.unsqueeze(x_in, 0)
    x_img = torch.unsqueeze(x_img, 0)
    y_target = torch.unsqueeze(y_target, 0)

    x_in = torch.unsqueeze(x_in, 1)
    x_img = torch.unsqueeze(x_img, 1)
    y_target = torch.unsqueeze(y_target, 1)

    x_img = x_img.clone().detach().to(device=device)
    x_in = x_in.clone().detach().to(device=device)
    y_target = y_target.clone().detach().to(device=device)

    [pred, loss_layers_sym, loss_st] = model(x_img, x_in, epoch)
    
    return pred[0,:,0:Nx,:].cpu().squeeze().detach(), x_img[0,:,0:Nx,:].cpu().squeeze().detach(), y_target.cpu().squeeze().detach(), snr_dir


def plot_comp(model, sample_ids, snr_ids, epoch, root_dir):
    col_no = len(sample_ids)
    fig, axs = plt.subplots(3, col_no)
    fig.set_figheight(1500*px)
    fig.set_figwidth(6000*px)
    
    for i in range(col_no):
        img, x_img, y_target, snr_dir = eval_model(fista_net, sample_ids[i], snr_ids[i], epoch, root_dir)
        #axs[0, i].set_title(snr_dir)
        axs[0, i].imshow(x_img)
        axs[0, i].get_xaxis().set_visible(False)
        axs[0, i].get_yaxis().set_visible(False)
        axs[1, i].imshow(y_target)
        axs[1, i].get_xaxis().set_visible(False)
        axs[1, i].get_yaxis().set_visible(False)
        axs[2, i].imshow(img)
        axs[2, i].get_xaxis().set_visible(False)
        axs[2, i].get_yaxis().set_visible(False)
    
    plt.savefig('./test.png')
    plt.clf()
    plt.close()


def eval_test(model, sample_id, snr_id, epoch, root_dir):
    img, x_img, y_target, snr_dir = eval_model(fista_net, sample_id, snr_id, epoch, root_dir)
    
    mdic = {"admm": admm_x_imgs, "fistanet": fista_net_preds}
    savemat("./test_vws/SNR_%d_ADMM_FISTA-Net.mat" % (snr_index), mdic)
    
    fista_net_preds.append(pred[:, :, :self.Nx, self.padding:-self.padding])
    admm_x_imgs.append(x_img[:, :, :self.Nx, self.padding:-self.padding])
    
    print(x_img)
    
    #A = np.double(A)
    #out = np.zeros(A.shape, np.double)
    #normalized = cv2.normalize(A, out, 1.0, 0.0, cv2.NORM_MINMAX)
    


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    device = 'cuda'

    J_ndt = np.loadtxt(pjoin(data_dir, "kernel_abel_matrix.csv"), delimiter=",", dtype=float)
    J_ndt = torch.from_numpy(J_ndt[:64,:])
    J_ndt = J_ndt.clone().detach().to(device=device)

    DMts = []

    mask = MatMask(64)
    mask = torch.from_numpy(mask)
    mask = mask.clone().detach()

    fista_net = FISTANet(LayerNo, FeatureNo, J_ndt, DMts, mask)
    fista_net = fista_net.to(device)

    snr_ids = [9, 6, 2, 1]
    sample_ids = [236, 236, 236, 236]
    epoch = 45

    plot_comp(fista_net, sample_ids, snr_ids, epoch, root_dir)
    #eval_test(fista_net, sample_ids[0], snr_ids[0], epoch, root_dir)

