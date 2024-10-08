# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:26:29 2020

@author: XIANG
"""
import torch.optim as optim
from M1LapReg import callLapReg
import torch
import torch.nn as nn
from os.path import dirname, join as pjoin
from collections import OrderedDict
import time
import numpy as np
from helpers import test_rescale, show_image_matrix
# from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from statistics import mean
from PIL import Image
import cv2


px = 1/plt.rcParams['figure.dpi']


def test_plot(x, file_name):
    Nx = 64
    batch_size = x.shape[0]
    fig, axs = plt.subplots(batch_size)
    fig.set_figheight(120*batch_size*px)
    fig.set_figwidth(640*px)
    if 1 == batch_size:
        axs.imshow(x[0,:,0:Nx,:].cpu().squeeze().detach())
    else:
        for i in range(batch_size):
            axs[i].imshow(x[i,:,0:Nx,:].cpu().squeeze().detach())
    plt.savefig(file_name)
    plt.clf()
    plt.close()
    #pass

def l1_loss(pred, target, l1_weight):
    """
    Compute L1 loss;
    l1_weigh default: 0.1
    """
    err = torch.mean(torch.abs(pred - target))
    err = l1_weight * err
    return err

def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss. 0.01 default.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    w_variance = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
    h_variance = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
    loss = tv_weight * (h_variance + w_variance)
    return loss


class Solver(object):
    def __init__(self, model, data_loader, val_loader, snr_nr, batch_size, args, test_data, test_images=None):
        assert args.model_name in ['FBPConv', 'ISTANet', 'FISTANet']

        self.model_name = args.model_name
        self.model = model
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.data_dir = args.data_dir
        self.num_epochs = args.num_epochs
        self.start_epoch = args.start_epoch
        self.lr = args.lr
        
        self.snr_nr = snr_nr
        self.batch_size = batch_size
        
        self.lr_dec_after = args.lr_dec_after
        self.lr_dec_every = args.lr_dec_every
        
        self.train_losses = []
        self.avg_val_losses_per_snr = []

        # NDT arguments
        self.padding = args.padding
        self.Nx = args.Nx

        if self.model_name == 'FISTANet':
            # set different lr for regularization weights and network weights
            self.optimizer = optim.Adam([
            {'params': self.model.fcs.parameters()},
            {'params': self.model.w_theta, 'lr': 0.001},
            {'params': self.model.b_theta, 'lr': 0.001},
            {'params': self.model.w_mu, 'lr': 0.001},
            {'params': self.model.b_mu, 'lr': 0.001},
            {'params': self.model.w_rho, 'lr': 0.001},
            {'params': self.model.b_rho, 'lr': 0.001}],
            lr=self.lr, weight_decay=0.001)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.001)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)  # step-wise

        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu
        self.device = args.device
        self.log_interval = args.log_interval
        self.test_epoch = args.test_epoch
        self.test_data = test_data
        self.test_images = test_images
        self.train_loss = nn.MSELoss()

#    def save_model(self, iter_):
#        if not os.path.exists(self.save_path):
#            os.makedirs(self.save_path)
#        f = pjoin(self.save_path, 'epoch_{}.ckpt'.format(iter_))
#        torch.save(self.model.state_dict(), f)

#    def load_model(self, iter_):
#        f = pjoin(self.save_path, 'epoch_{}.ckpt'.format(iter_))
#        if self.multi_gpu:
#            state_d = OrderedDict()
#            for k, v in torch.load(f):
#                n = k[7:]
#                state_d[n] = v
#            self.model.load_state_dict(state_d)
#        else:
#            self.model.load_state_dict(torch.load(f))
    
    def save_model(self, iter_):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        f = pjoin(self.save_path, 'epoch_{}.ckpt'.format(iter_))
        checkpoint = { 
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'avg_val_losses_per_snr': self.avg_val_losses_per_snr}
        torch.save(checkpoint, f)

    def load_model(self, iter_):
        f = pjoin(self.save_path, 'epoch_{}.ckpt'.format(iter_))
        checkpoint = torch.load(f,  map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # decrease learning rate per X epochs after a set number of epochs
        if iter_ >= self.lr_dec_after:
            if not (iter_ - self.lr_dec_after):
                self.scheduler.step()
            elif not ((iter_ - self.lr_dec_after) % self.lr_dec_every):
                self.scheduler.step()

    def train(self):
        start_time = time.time()
        # set up Tensorboard
        # writer = SummaryWriter('runs/'+self.model_name)

        for epoch in range(1 + self.start_epoch, self.num_epochs + 1 + self.start_epoch):
            print('Training epoch %d...' % epoch)
            
            self.train_losses = []

            self.model.train(True)

            for batch_idx, ((x_in, x_img), y_target) in enumerate(self.data_loader):

                # measured vector (104*1); add channels
                # CIKK: vector b (16) --\/
                x_in = torch.unsqueeze(x_in, 1)

                # initial image from one-step inversion
                # CIKK: initialization (16) --\/
                # x_img = callLapReg(data_dir=self.data_dir, y_test=x_in)
                x_img = torch.unsqueeze(x_img, 1)

                # target image (64*64)
                y_target = torch.unsqueeze(y_target, 1)

                if epoch == 1 and batch_idx == 1:
                    show_image_matrix('./figures/X0.png', [x_img, y_target],
                                      titles=['X0 FBP', 'y_target'],
                                      indices=slice(0, 30))

                x_img = x_img.clone().detach().to(device=self.device)
                x_in = x_in.clone().detach().to(device=self.device)
                y_target = y_target.clone().detach().to(device=self.device)

                if self.model_name == 'FBPConv':

                    pred = self.model(x_img)
                    loss = self.train_loss(pred, y_target) + l1_loss(pred, y_target, 0.1)

                # predict and compute losses
                if self.model_name == 'ISTANet':
                    [pred, loss_sym] = self.model(x_img, x_in)
                    loss_discrepancy = self.train_loss(pred, y_target) + l1_loss(pred, y_target, 0.1)
                    loss_constraint = 0
                    for k, _ in enumerate(loss_sym, 0):
                        loss_constraint += torch.mean(torch.pow(loss_sym[k], 2))

                    loss = loss_discrepancy + 0.01 * loss_constraint

                if self.model_name == 'FISTANet':
                    [pred, loss_layers_sym, loss_st] = self.model(x_img, x_in, epoch)   # forward
                    
                    # [pred, loss_layers_sym, loss_st] = self.model(x_img, x_in)   # forward
                    test_plot(pred, '.\\testing\\pred_%d.png' % (epoch))
                    test_plot(y_target, '.\\testing\\target.png')

                    # Compute loss, data consistency and regularizer constraints
                    # NEW: csak az első paraméter db sorra (levágni az utolsó sorokat)
                    loss_discrepancy_1 = self.train_loss(pred[:, :, :self.Nx, self.padding:-self.padding],
                                                         y_target[:, :, :self.Nx, self.padding:-self.padding])  # +\
                    loss_discrepancy_2 = l1_loss(pred[:, :, :self.Nx, self.padding:-self.padding],
                                                 y_target[:, :, :self.Nx, self.padding:-self.padding], 0.1)
                    loss_discrepancy = loss_discrepancy_1 + loss_discrepancy_2
                    loss_constraint = 0
                    for k, _ in enumerate(loss_layers_sym, 0):
                        loss_constraint += torch.mean(torch.pow(loss_layers_sym[k], 2))

                    sparsity_constraint = 0
                    for k, _ in enumerate(loss_st, 0):
                        sparsity_constraint += torch.mean(torch.abs(loss_st[k]))

                    # loss = loss_discrepancy + gamma * loss_constraint
                    # CIKK: (14) --\/
                    loss = loss_discrepancy + 0.01 * loss_constraint + 0.001 * sparsity_constraint

                self.model.zero_grad()
                self.optimizer.zero_grad()

                # backpropagate the gradients
                loss.backward()
                self.optimizer.step()
                self.train_losses.append(loss.item())

                # print processes
                if batch_idx % self.log_interval == 0:
                    # writer.add_scalar('training loss', loss.data, epoch * len(self.data_loader) + batch_idx)

                    print()
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}\tLearning Rate (w_theta): {:.6f}\t TIME:{:.1f}s'
                          ''.format(epoch, batch_idx * len(x_in),
                                    len(self.data_loader.dataset),
                                    100. * batch_idx / len(self.data_loader),
                                    loss.data,
                                    self.optimizer.param_groups[0]["lr"],
                                    time.time() - start_time))

                    print('\t\t\t\tDisc: {:.6f} \tConst: {:.6f}\t\tSpars: {:.6f}'
                          ''.format(loss_discrepancy.data, 0.01*loss_constraint.data, 0.001 * sparsity_constraint.data))


                    # print weight values of model
                    if self.model_name == 'FISTANet':
                        print('\t TVw: {:.6f} | TVb: {:.6f} | GSw: {:.6f} | GSb: {:.6f} | TSUw: {:.6f} | TSUb: {:.6f}'
                              ''.format(self.model.w_theta.item(), self.model.b_theta.item(), self.model.w_mu.item(), self.model.b_mu.item(), self.model.w_rho.item(), self.model.b_rho.item()))
 
#                        print("Threshold value w: {}".format(self.model.w_theta))
#                        print("Threshold value b: {}".format(self.model.b_theta))
#                        print("Gradient step w: {}".format(self.model.w_mu))
#                        print("Gradient step b: {}".format(self.model.b_mu))
#                        print("Two step update w: {}".format(self.model.w_rho))
#                        print("Two step update b: {}".format(self.model.b_rho))
            
            print('Validating epoch %d...' % epoch)
            val_size = len(self.val_loader) * self.batch_size
            elems_per_snr = val_size // self.snr_nr
            batches_per_snr = elems_per_snr // self.batch_size
            self.avg_val_losses_per_snr = []
            self.model.eval()
            with torch.no_grad():
                for batch_idy, ((x_in, x_img), y_target) in enumerate(self.val_loader):
                    # compute which batch of which SNR we are at
                    snr_index = batch_idy // batches_per_snr
                    snr_batch_index = batch_idy % batches_per_snr
                    next_snr_batch_index = (batch_idy + 1) % batches_per_snr
                    
                    # empty per SNR list if at new SNR
                    if not snr_batch_index:
                        snr_val_losses = []
                    
                    x_in = torch.unsqueeze(x_in, 1)
                    x_img = torch.unsqueeze(x_img, 1)
                    y_target = torch.unsqueeze(y_target, 1)
                    
                    x_img = x_img.clone().detach().to(device=self.device)
                    x_in = x_in.clone().detach().to(device=self.device)
                    y_target = y_target.clone().detach().to(device=self.device)
                    
                    [pred, loss_layers_sym, loss_st] = self.model(x_img, x_in, epoch)   # forward
                    
                    # plot validation batch
                    test_plot(pred, '.\\testing\\valid_ep%d_snr%d_btch%d.png' % (epoch, snr_index, snr_batch_index))

                    # Compute loss, data consistency and regularizer constraints
                    loss_discrepancy_1 = self.train_loss(pred[:, :, :self.Nx, self.padding:-self.padding],
                                                         y_target[:, :, :self.Nx, self.padding:-self.padding])  # +\
                    loss_discrepancy_2 = l1_loss(pred[:, :, :self.Nx, self.padding:-self.padding],
                                                 y_target[:, :, :self.Nx, self.padding:-self.padding], 0.1)
                    loss_discrepancy = loss_discrepancy_1 + loss_discrepancy_2
                    loss_constraint = 0
                    for k, _ in enumerate(loss_layers_sym, 0):
                        loss_constraint += torch.mean(torch.pow(loss_layers_sym[k], 2))
                    sparsity_constraint = 0
                    for k, _ in enumerate(loss_st, 0):
                        sparsity_constraint += torch.mean(torch.abs(loss_st[k]))
                    loss = loss_discrepancy + 0.01 * loss_constraint + 0.001 * sparsity_constraint
                    
                    # add batch loss to per SNR list
                    snr_val_losses.append(loss.item())
                    
                    # calculate and save average SNR loss if at last batch of SNR
                    if not next_snr_batch_index:
                        self.avg_val_losses_per_snr.append(mean(snr_val_losses))
            
            print('-------------------------------------------')
            print('Epoch statistics:')
            print('Average training loss:', mean(self.train_losses))
            print('Average validation losses per SNR:', self.avg_val_losses_per_snr)
            
            save_every = 1        # save model ever N-th epoch
            if not (epoch % save_every) and epoch > 0:
                self.save_model(epoch);
            
            # decrease learning rate per X epochs after a set number of epochs
            if epoch >= self.lr_dec_after:
                if not (epoch - self.lr_dec_after):
                    self.scheduler.step()
                elif not ((epoch - self.lr_dec_after) % self.lr_dec_every):
                    self.scheduler.step()


    def test(self):
        self.load_model(self.test_epoch)
        self.model.eval()

        with torch.no_grad():
            # Must use the sample test dataset!!!
            # callLapReg(data_dir=self.data_dir, y_test=self.test_data)
            x_test_in = self.test_data[0]
            #x_test_in = torch.unsqueeze(x_test_in, 1)
            x_test_in = x_test_in.clone().detach().to(device=self.device)

            if self.model_name == "ISTANet":
                x_test_img = self.test_data[1]
                #x_test_img = torch.unsqueeze(x_test_img, 1)
                x_test_img = x_test_img.clone().detach().to(device=self.device)
                [test_res, _] = self.model(x_test_img, x_test_in, 999)

            elif self.model_name == 'FISTANet':
                x_test_img = self.test_data[1]
                #x_test_img = torch.unsqueeze(x_test_img, 1)
                x_test_img = x_test_img.clone().detach().to(device=self.device)
                [test_res, _, _] = self.model(x_test_img, x_test_in, 999)

            else:
                # other nets needs to do one-step inversion
                test_res = self.model(x_test_in)

        return test_res
    
    def test_MSE(self, test_loader, epoch):
        SNRS = [-10, -20, 0, 10, 20, 30, 40, 50, 60, 70]
        
        start_time = time.time()

        print('Testing epoch %d...' % epoch)
        
        data_dir = "../data/NDTData/"
        data_dir = "D:\\USERS\\galiger.gergo\\for_vw_benchmarking\\RotatedData_filt_2000\\Deg_45"
        fle = pjoin(data_dir, 'abel_transf_constants.mat')
        mat_file = loadmat(fle)
        invabel = mat_file['invabel']
        normKabel = mat_file['normKabel']
        invabels = np.array([[invabel] for i in range(self.batch_size)])
        
        test_size = len(test_loader) * self.batch_size
        elems_per_snr = test_size // self.snr_nr
        batches_per_snr = elems_per_snr // self.batch_size
        avg_test_losses_per_snr_FISTANet = []
        avg_test_losses_per_snr_ADMM = []

        self.model.eval()
        with torch.no_grad():
            for batch_idx, ((x_in, x_img), y_target) in enumerate(test_loader):
                if batch_idx < 0:
                    print('Skipping batch %d...' % batch_idx)
                    snr_test_losses_FISTANet = []
                    snr_test_losses_ADMM = []
                    fista_net_preds = []
                    admm_x_imgs = []
                    continue
                
                if len(x_in) < self.batch_size:
                    invabels = np.array([[invabel] for i in range(len(x_in))])
                
                # compute which batch of which SNR we are at
                #snr_index = batch_idx // batches_per_snr
                #snr_batch_index = batch_idx % batches_per_snr
                #next_snr_batch_index = (batch_idx + 1) % batches_per_snr
                
                # empty per SNR list if at new SNR
                #if not snr_batch_index:
                    #snr_test_losses_FISTANet = []
                    #snr_test_losses_ADMM = []
                    #fista_net_preds = []
                    #admm_x_imgs = []
                    #path = 'C:/Users/galiger.gergo/Desktop/PlotData/FKMIG_data/MAT/SNR_%d' % SNRS[snr_index]
                    #path = 'D:/USERS/galiger.gergo/FKMIG_data_abel_space/MAT/SNR_%d' % SNRS[snr_index]
                    #path = 'D:/USERS/galiger.gergo/UNetTrainData/MAT/SNR_%d' % SNRS[snr_index]
                    #if not os.path.exists(path):
                    #   os.makedirs(path)
                    #path2 = 'D:/USERS/galiger.gergo/FKMIG_data_abel_space/MAT/GroundTruth'
                    #if not os.path.exists(path2):
                    #   os.makedirs(path2)
                    #path3 = 'C:\\Users\\galiger.gergo\\Desktop\\ThermUNet-master\\data\\hybrid\\train\\admm\\SNR_%d' % SNRS[snr_index]
                    #if not os.path.exists(path3):
                    #   os.makedirs(path3)
                    #path4 = 'C:\\Users\\galiger.gergo\\Desktop\\ThermUNet-master\\data\\hybrid\\train\\fistanet\\SNR_%d' % SNRS[snr_index]
                    #if not os.path.exists(path4):
                    #   os.makedirs(path4)
                path = "D:\\USERS\\galiger.gergo\\for_vw_benchmarking\\RotatedData_filt_2000\\Deg_45\\FISTANet"

                x_in = torch.unsqueeze(x_in, 1)
                x_img = torch.unsqueeze(x_img, 1)
                y_target = torch.unsqueeze(y_target, 1)

                x_img = x_img.clone().detach().to(device=self.device)
                x_in = x_in.clone().detach().to(device=self.device)
                y_target = y_target.clone().detach().to(device=self.device)

                [pred, loss_layers_sym, loss_st] = self.model(x_img, x_in, epoch)

                # transform back from abel space to virtual wave space
                preds = pred.cpu().detach().numpy()
                x_imgs = x_img.cpu().detach().numpy()
                y_targets = y_target.cpu().detach().numpy()
                shp = y_targets.shape
                shp = (shp[0], shp[1], invabels.shape[2], shp[3])
                temp = np.copy(y_targets)
                y_targets = np.zeros(shp)
                y_targets[:shp[0], :shp[1], :temp.shape[2], :shp[3]] = temp
                preds_virt_space = np.pi / 2.0 * np.matmul(invabels, preds) / normKabel
                x_imgs_virt_space = np.pi / 2.0 * np.matmul(invabels, x_imgs) / normKabel
                y_targets_virt_space = np.pi / 2.0 * np.matmul(invabels, y_targets) / normKabel
                #for i in range(self.batch_size):
                for i in range(len(x_img)):
                    mdic = {"admm_virt_space": x_imgs_virt_space[i][0], "fistanet_virt_space": preds_virt_space[i][0]}
                    savemat(pjoin(path, 'MAT', 'virtualwave_%d.mat' % (batch_idx * self.batch_size + i + 1)), mdic)
                    #mdic = {"admm_virt_space": x_imgs_virt_space[i][0], "fistanet_virt_space": preds_virt_space[i][0]}
                    #mdic2 = {"y_target_abel_space": y_targets[i][0]}
                    #savemat('C:/Users/galiger.gergo/Desktop/PlotData/FKMIG_data/MAT/SNR_%d/virtualwave_%d.mat' % (SNRS[snr_index], snr_batch_index * self.batch_size + i + 1), mdic)
                    #savemat('D:/USERS/galiger.gergo/FKMIG_data_abel_space/MAT/GroundTruth/virtualwave_%d.mat' % (snr_batch_index * self.batch_size + i + 1), mdic2)
                    #savemat('D:/USERS/galiger.gergo/FKMIG_data_abel_space/MAT/SNR_%d/virtualwave_%d.mat' % (SNRS[snr_index], snr_batch_index * self.batch_size + i + 1), mdic)
                    #savemat('D:/USERS/galiger.gergo/UNetTrainData/MAT/SNR_%d/virtualwave_%d.mat' % (SNRS[snr_index], snr_batch_index * self.batch_size + i + 1), mdic)
                    #img1 = Image.fromarray(x_imgs_virt_space[i, 0, :self.Nx, self.padding:-self.padding])
                    #img1 = img1.convert("L")
                    #img1.save('C:\\Users\\galiger.gergo\\Desktop\\ThermUNet-master\\data\\hybrid\\train\\admm\\SNR_%d\\virtualwave_%d.png' % (SNRS[snr_index], snr_batch_index * self.batch_size + i + 1))
                    #img1 = Image.fromarray(x_imgs[i, 0, :48])
                    #img1 = img1.convert("L")
                    #img1.save(pjoin(path, 'virtualwave_%d.png' % (batch_idx * self.batch_size + i + 1)))
                    #img2 = Image.fromarray(preds_virt_space[i, 0, :self.Nx, self.padding:-self.padding])
                    #img2 = img2.convert("L")
                    #img2.save('C:\\Users\\galiger.gergo\\Desktop\\ThermUNet-master\\data\\hybrid\\train\\fistanet\\SNR_%d\\virtualwave_%d.png' % (SNRS[snr_index], snr_batch_index * self.batch_size + i + 1))
                preds_virt_space = torch.Tensor(preds_virt_space).clone().detach().to(device=self.device)
                x_imgs_virt_space = torch.Tensor(x_imgs_virt_space).clone().detach().to(device=self.device)
                y_targets_virt_space = torch.Tensor(y_targets_virt_space).clone().detach().to(device=self.device)
                
                
                #loss_MSE_FISTANet = self.train_loss(preds_virt_space[:, :, :self.Nx, self.padding:-self.padding],
                #                                    y_targets_virt_space[:, :, :self.Nx, self.padding:-self.padding])
                #loss_MSE_ADMM = self.train_loss(x_imgs_virt_space[:, :, :self.Nx, self.padding:-self.padding],
                #                                y_targets_virt_space[:, :, :self.Nx, self.padding:-self.padding])

                # add batch loss to per SNR list
                #snr_test_losses_FISTANet.append(loss_MSE_FISTANet.item())
                #snr_test_losses_ADMM.append(loss_MSE_ADMM.item())
                #Apred = preds_virt_space[:, :, :self.Nx, self.padding:-self.padding].cpu().squeeze().detach().numpy()
                #Aximg = x_imgs_virt_space[:, :, :self.Nx, self.padding:-self.padding].cpu().squeeze().detach().numpy()
                #for b_ind in range(Apred.shape[0]):
                    #fista_net_preds.append(Apred[b_ind])
                    #admm_x_imgs.append(Aximg[b_ind])
                
                # calculate and save average SNR loss if at last batch of SNR
                #if not next_snr_batch_index:
                    #avg_test_losses_per_snr_FISTANet.append(mean(snr_test_losses_FISTANet))
                    #avg_test_losses_per_snr_ADMM.append(mean(snr_test_losses_ADMM))
                    #mdic = {"admm": admm_x_imgs, "fistanet": fista_net_preds}
                    #savemat("./test_vws/SNR_%d_ADMM_FISTA-Net.mat" % (snr_index), mdic)

                #print('Test Epoch: {} [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}\tTIME:{:.1f}s'
                      #''.format(epoch, batch_idx * len(x_in),
                                #len(test_loader.dataset),
                                #100. * batch_idx / len(test_loader),
                                #loss_MSE_FISTANet.data,
                                #time.time() - start_time))
                print('Test Epoch: {} [{}/{} ({:.0f}%)]\tTIME:{:.1f}s'
                      ''.format(epoch, batch_idx * len(x_in),
                                len(test_loader.dataset),
                                100. * batch_idx / len(test_loader),
                                time.time() - start_time))

        # switch -10 and -20 SNR order
        #temp = avg_test_losses_per_snr_FISTANet[0]
        #avg_test_losses_per_snr_FISTANet[0] = avg_test_losses_per_snr_FISTANet[1]
        #avg_test_losses_per_snr_FISTANet[1] = temp
        #temp = avg_test_losses_per_snr_ADMM[0]
        #avg_test_losses_per_snr_ADMM[0] = avg_test_losses_per_snr_ADMM[1]
        #avg_test_losses_per_snr_ADMM[1] = temp

        return [], []
        #return avg_test_losses_per_snr_FISTANet, avg_test_losses_per_snr_ADMM

