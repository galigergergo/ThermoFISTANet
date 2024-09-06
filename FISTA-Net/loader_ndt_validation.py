# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 13:51:58 2022

DataLoader of NDT multi-level (only input, no target) data.

@author: Galiger Gergo

"""
from torch.utils.data import Dataset, DataLoader
import numpy as np
from os.path import join as pjoin
from torch.utils.data.sampler import SubsetRandomSampler
from scipy.io import loadmat
import os


class NDTDataset(Dataset):
    """Prepare NDT Dataset"""

    def __init__(self, mode, data_dir, nr_diff_imgs, nr_diff_imgs_train, nr_diff_imgs_val, transform=None):
        # training, test and validate dataset;
        assert mode in ['train', 'test']

        np.warnings.filterwarnings('ignore',
                                   category=np.VisibleDeprecationWarning)
        
        self.mode = mode
        
        self.inp_dir = pjoin(data_dir, 'VirtualWaves')
        
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
        fle = pjoin(sample_dir, 'virtualwave_%d.mat' % (sample_id + 1))
        mat_file = loadmat(fle)
        input_m = (mat_file['T_noisy'], mat_file['T_virt_in_abelspace'])
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
                             shuffle=True, num_workers=num_workers)
    return data_loader


def DataSplit(root_dir, nr_diff_imgs, batch_size=128, validation_split=0.1, test_split=0.1, transform=None):
    train_split = 1.0 - test_split - validation_split
    
    nr_diff_imgs_train = int(np.floor(nr_diff_imgs * (1 - validation_split - test_split)))
    nr_diff_imgs_val = int(np.floor(nr_diff_imgs * validation_split))
    
    #nr_diff_imgs_train = 8000
    
    shuffle_dataset = True
    random_seed = 42
    dataset = NDTDataset(mode='train', data_dir=root_dir, nr_diff_imgs=nr_diff_imgs, nr_diff_imgs_train=nr_diff_imgs_train, nr_diff_imgs_val=nr_diff_imgs_val, transform=transform)
    test_loader = get_loader(nr_diff_imgs, nr_diff_imgs_train, nr_diff_imgs_val, mode='test', data_dir=root_dir,
                             batch_size=batch_size, transform=transform)

    # Creating data indices for training and validation splits:
    train_size = len(dataset)
    train_val_size = int(np.floor((train_split + validation_split) / train_split * train_size))
    dataset_size = int(np.floor(1.0 / train_split * train_size))
    
    #train_val_size = 90000
    #dataset_size = 90000
    
    
    indices = list(range(train_val_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    split = int(np.floor(validation_split * dataset_size))
    #split = 10000
    train_indices, val_indices = indices[split:], indices[:split]    
    print('Total valid data size: ', split)
    print('Total train data size: ', train_size)
    
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=valid_sampler)

    test_loader = train_loader
    return train_loader, val_loader, test_loader
