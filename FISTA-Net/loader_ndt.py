# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 13:51:58 2022

DataLoader of NDT multi-level (only input, no target) data.

@author: Galiger Gergo

"""
from torch.utils.data import Dataset, DataLoader
import numpy as np
from os.path import join as pjoin
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from scipy.io import loadmat
import os
import random


# sample validation data per SNR
class CustomSubsetRandomSampler(Sampler):
  def __init__(self, indices, snr_nr, batch_size):
    self.indices = indices
    self.snr_nr = snr_nr
    self.per_snr_indices = []
    self.elems_per_snr = len(self.indices) // self.snr_nr
    # one SNR data has to fit perfectly in N number of batches
    #assert self.elems_per_snr % batch_size == 0
    self.build_per_snr_list()
    
  def build_per_snr_list(self):
    for i in range(self.snr_nr):
        self.per_snr_indices += [j * self.snr_nr + i for j in range(self.elems_per_snr)]
    
  def __iter__(self):
    return iter(self.per_snr_indices)

  def __len__(self):
    return len(self.indices)


class NDTDataset(Dataset):
    """Prepare NDT Dataset"""

    def __init__(self, mode, data_dir, nr_diff_imgs, nr_diff_imgs_train, nr_diff_imgs_val, transform=None):
        # training, test and validate dataset;
        assert mode in ['train', 'test']

        np.warnings.filterwarnings('ignore',
                                   category=np.VisibleDeprecationWarning)
        
        self.mode = mode
        
        #self.inp_dir = pjoin(data_dir, 'VirtualWaves')
        self.inp_dir = data_dir
        
        self.nr_diff_imgs = nr_diff_imgs
        self.nr_diff_imgs_train = nr_diff_imgs_train
        self.nr_diff_imgs_val = nr_diff_imgs_val

        self.transform = transform
        self.input_ = []
        self.target_ = []
        
        self.snr_nr = len(os.listdir(self.inp_dir))
        
        if mode == 'train':
            self.nr_samples = self.nr_diff_imgs_train * self.snr_nr
        else:
            self.nr_samples = (self.nr_diff_imgs - self.nr_diff_imgs_train - self.nr_diff_imgs_val) * self.snr_nr
        

    def __len__(self):
        return self.nr_samples

    def __getitem__(self, idx):
        # separate snr and sample_id from idx
        snr_id = idx % self.snr_nr
        sample_id = idx // self.snr_nr
        
        # build directory path to sample
        snr_dir = os.listdir(self.inp_dir)[snr_id]
        sample_dir = pjoin(self.inp_dir, snr_dir)
        sample_dir = pjoin(sample_dir, 'MAT')
        
        # read mat file
        if self.mode == 'test':
            sample_id += self.nr_diff_imgs_train + self.nr_diff_imgs_val
        #fle = pjoin(sample_dir, 'virtualwave_%d.mat' % (sample_id + 1))
        fle = pjoin(self.inp_dir, 'virtualwave_%d.mat' % (idx + 1))
        mat_file = loadmat(fle)
        input_m = (mat_file['T_noisy'][:48, :], mat_file['T_virt_in_abelspace'])
        target_img = mat_file['T_virtual_abel']
        
        # transform the input tensor into required formats
        if self.transform:
            input_m = self.transform(input_m)

        return (input_m, target_img)


def get_loader(nr_diff_imgs, nr_diff_imgs_train, nr_diff_imgs_val, mode='train', data_dir=None, transform=None, batch_size=128,
               num_workers=4):
    dataset_ = NDTDataset(mode=mode, data_dir=data_dir, nr_diff_imgs=nr_diff_imgs, nr_diff_imgs_train=nr_diff_imgs_train, nr_diff_imgs_val=nr_diff_imgs_val, transform=transform)
    print('Total', mode, ' data size: ', len(dataset_))
    data_loader = DataLoader(dataset=dataset_, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    return data_loader


def DataSplit(root_dir, nr_diff_imgs, batch_size=128, validation_split=0.1, test_split=0.1, transform=None):
    train_split = 1.0 - test_split - validation_split
    
    nr_diff_imgs_train = int(np.floor(nr_diff_imgs * train_split))
    nr_diff_imgs_val = int(np.floor(nr_diff_imgs * validation_split))
    
    #shuffle_dataset = True
    #random_seed = 42
    dataset = NDTDataset(mode='train', data_dir=root_dir, nr_diff_imgs=nr_diff_imgs, nr_diff_imgs_train=nr_diff_imgs_train, nr_diff_imgs_val=nr_diff_imgs_val, transform=transform)
    dataset_test = NDTDataset(mode='test', data_dir=root_dir, nr_diff_imgs=nr_diff_imgs, nr_diff_imgs_train=nr_diff_imgs_train, nr_diff_imgs_val=nr_diff_imgs_val, transform=transform)
    # test_loader = get_loader(nr_diff_imgs, nr_diff_imgs_train, nr_diff_imgs_val, mode='test', data_dir=root_dir,
    #                          batch_size=batch_size, transform=transform)

    # Creating data indices for training and validation splits:
    train_size = len(dataset)
    train_val_size = int(np.floor((train_split + validation_split) / train_split * train_size))
    dataset_size = int(np.floor(1.0 / train_split * train_size))    
    
    indices = list(range(train_val_size))
    #if shuffle_dataset:
    #    np.random.seed(random_seed)
    #    np.random.shuffle(indices)
    split = train_size
    #split = 0
    train_indices, val_indices = indices[:split], indices[split:]
    test_indices = list(range(dataset_size))[train_val_size:]

    print('Total test  data size: ', len(test_indices))
    print('Total valid data size: ', len(val_indices))
    print('Total train data size: ', len(train_indices))
    
    # one SNR data has to fit perfectly in N number of batches for validation
    val_elems_per_snr = len(val_indices) // dataset.snr_nr
    assert val_elems_per_snr % batch_size == 0
        
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    train_sampler = CustomSubsetRandomSampler(train_indices, dataset.snr_nr, batch_size)
    
    valid_sampler = CustomSubsetRandomSampler(val_indices, dataset.snr_nr, batch_size)
    test_sampler = CustomSubsetRandomSampler(test_indices, dataset.snr_nr, batch_size)

    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=valid_sampler)
    test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size,
                            sampler=test_sampler)

    return train_loader, val_loader, test_loader, dataset.snr_nr
