import torch
from os.path import dirname, join as pjoin
from statistics import mean
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.io import loadmat


def swap_first_two(lst):
    temp = lst[0]
    lst[0] = lst[1]
    lst[1] = temp
    return lst


train_data_dir = 'D:\\USERS\\galiger.gergo\\TrainData\\10000'
SNRS = os.listdir(pjoin(train_data_dir, 'VirtualWaves'))
SNRS = swap_first_two(SNRS)

y_labels = {
    'l_r': 'Learning Rate',
    'train_losses': 'Train Loss (Disc + Const + Spars)',
    'avg_val_losses_per_snr': 'Validation Loss (Disc + Const + Spars)'
    }
titles = {
    'l_r': 'Learning Rate',
    'train_losses': 'Train Loss'
    }

save_path = './models_copy/FISTANet/'

curvelet_path = '../data/NDTData/MSE_per_SNR_curvelet_ADMM_test.mat'

test_loss_path = './test_losses'

plot_dir = './plots'

epochs = (1, 50)


def get_train_epoch_data(epoch, params):    # params in ['lr', 'train_losses', 'avg_val_losses_per_snr']
    f = pjoin(save_path, 'epoch_{}.ckpt'.format(epoch))
    checkpoint = torch.load(f)
    res = dict()
    for param in params:
        if param == 'lr':
            res[param] = checkpoint['optimizer']['param_groups'][0]['lr']
        elif param == 'avg_val_losses_per_snr':
            res[param] = swap_first_two(checkpoint[param])
        else:
            res[param] = checkpoint[param]
    return res


def get_train_epochs_data(epochs, params):    # params in ['lr', 'train_losses', 'avg_val_losses_per_snr']
    res = dict()
    for epoch in range(epochs[0], epochs[1] + 1):
        f = pjoin(save_path, 'epoch_{}.ckpt'.format(epoch))
        checkpoint = torch.load(f)
        res_ep = dict()
        for param in params:
            if param == 'lr':
                res_ep[param] = checkpoint['optimizer']['param_groups'][0]['lr']
            elif param == 'avg_val_losses_per_snr':
                res_ep[param] = swap_first_two(checkpoint[param])
            else:
                res_ep[param] = checkpoint[param]
        res[epoch] = res_ep
    return res


def plot_param_per_epoch(epochs, param, model_name):
    assert param in ['lr', 'train_losses']
    data = get_train_epochs_data(epochs, [param])
    x_epochs = list(data.keys())
    if param == 'train_losses':
        y_values = [mean(data[k][param]) for k in x_epochs]
    else:
        y_values = [data[k][param] for k in x_epochs]
    plt.figure()
    plt.plot(x_epochs, y_values, '-or', label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel(y_labels[param])
    plt.title('%s values per epoch' % titles[param])
    plt.legend()
    plot_fig_dir = pjoin(plot_dir, model_name)
    plot_fig_dir = pjoin(plot_fig_dir, param)
    if not os.path.exists(plot_fig_dir):
        os.makedirs(plot_fig_dir)
    plt.savefig(pjoin(plot_fig_dir, '%s_per_epoch_%d-%d.png' % (param, epochs[0], epochs[1])))
    plt.close()


def plot_avg_val_loss_per_epoch_per_snr(epochs, model_name):
    param = 'avg_val_losses_per_snr'
    data = get_train_epochs_data(epochs, [param])
    x_epochs = list(data.keys())
    plot_fig_dir = pjoin(plot_dir, model_name)
    plot_fig_dir = pjoin(plot_fig_dir, 'val_losses/per_epoch/single_snr')
    if not os.path.exists(plot_fig_dir):
        os.makedirs(plot_fig_dir)
    for snr_id in range(len(data[epochs[0]][param])):
        y_values = [data[k][param][snr_id] for k in data]
        plt.figure()
        plt.ylim(10, 250)
        plt.plot(x_epochs, y_values, '-or', label=model_name)
        plt.xlabel('Epoch')
        plt.ylabel(y_labels[param])
        plt.title('Validation Loss values per epoch - %s'% SNRS[snr_id])
        plt.legend()
        plt.savefig(pjoin(plot_fig_dir, '%s_epochs_%d-%d.png' % (SNRS[snr_id], epochs[0], epochs[1])))
        plt.close()

    
def plot_avg_val_loss_per_epoch(epochs, snr_ids, model_name, limit_axis=True):
    param = 'avg_val_losses_per_snr'
    data = get_train_epochs_data(epochs, [param])
    x_epochs = list(data.keys())
    plot_fig_dir = pjoin(plot_dir, model_name)
    plot_fig_dir = pjoin(plot_fig_dir, 'val_losses/per_epoch/multi_snr')
    if not os.path.exists(plot_fig_dir):
        os.makedirs(plot_fig_dir)
    plt.figure()
    if limit_axis:
        plt.ylim(10, 250)
    for snr_id in snr_ids:
        y_values = [data[k][param][snr_id] for k in data]
        plt.plot(x_epochs, y_values, '-o', label=SNRS[snr_id])
    plt.xlabel('Epoch')
    plt.ylabel(y_labels[param])
    plt.title('Validation Loss values per epoch -- %s - %s'% (SNRS[snr_ids[0]], SNRS[snr_ids[-1]]))
    plt.legend()
    plt.savefig(pjoin(plot_fig_dir, '%s_%s_epochs_%d-%d_%d.png' % (SNRS[snr_ids[0]], SNRS[snr_ids[-1]], epochs[0], epochs[1], len(snr_ids))))
    plt.close()


def plot_avg_val_loss_per_epoch_avg(epochs, snr_ids, model_name, limit_axis=True):
    param = 'avg_val_losses_per_snr'
    data = get_train_epochs_data(epochs, [param])
    x_epochs = list(data.keys())
    plot_fig_dir = pjoin(plot_dir, model_name)
    plot_fig_dir = pjoin(plot_fig_dir, 'val_losses/per_epoch/avg')
    if not os.path.exists(plot_fig_dir):
        os.makedirs(plot_fig_dir)
    y_values = []
    plt.figure()
    if limit_axis:
        plt.ylim(10, 250)
    for snr_id in snr_ids:
        y_values.append([data[k][param][snr_id] for k in data])
    y_values = np.transpose(np.array(y_values))
    y_values = [mean(y_val_epoch) for y_val_epoch in y_values]
    plt.plot(x_epochs, y_values, '-o', label=SNRS[snr_id])
    plt.xlabel('Epoch')
    plt.ylabel(y_labels[param])
    plt.title('Validation Loss values per epoch -- %s - %s'% (SNRS[snr_ids[0]], SNRS[snr_ids[-1]]))
    plt.legend()
    plt.savefig(pjoin(plot_fig_dir, '%s_%s_epochs_%d-%d_avg.png' % (SNRS[snr_ids[0]], SNRS[snr_ids[-1]], epochs[0], epochs[1])))
    plt.close()


def plot_avg_val_loss_per_snr_per_epoch(epochs, snr_ids, model_name):
    param = 'avg_val_losses_per_snr'
    data = get_train_epochs_data(epochs, [param])
    x_snrs = [int(SNRS[id][4:]) for id in snr_ids]
    plot_fig_dir = pjoin(plot_dir, model_name)
    plot_fig_dir = pjoin(plot_fig_dir, 'val_losses/per_snr/single_epoch')
    if not os.path.exists(plot_fig_dir):
        os.makedirs(plot_fig_dir)
    for epoch in range(epochs[0], epochs[1] + 1):
        y_values = data[epoch][param]
        plt.figure()
        plt.ylim(10, 250)
        plt.plot(x_snrs, y_values, '-or', label=model_name)
        plt.xlabel('SNR')
        plt.ylabel(y_labels[param])
        plt.title('Validation Loss values per SNR - epoch %d'% epoch)
        plt.legend()
        plt.savefig(pjoin(plot_fig_dir, 'epoch-%d_%s-%s.png' % (epoch, SNRS[snr_ids[0]], SNRS[snr_ids[-1]])))
        plt.close()


def plot_train_and_avg_val_loss_per_epoch(epochs, snr_ids, model_name, limit_axis=True):
    param = 'avg_val_losses_per_snr'
    data = get_train_epochs_data(epochs, [param])
    param_t = 'train_losses'
    data_train = get_train_epochs_data(epochs, [param_t])
    x_epochs = list(data.keys())
    y_values_train = [mean(data_train[k][param_t]) for k in x_epochs]
    plot_fig_dir = pjoin(plot_dir, model_name)
    plot_fig_dir = pjoin(plot_fig_dir, 'train_val_losses/per_epoch/avg')
    if not os.path.exists(plot_fig_dir):
        os.makedirs(plot_fig_dir)
    y_values = []
    plt.figure()
    if limit_axis:
        plt.ylim(10, 250)
    for snr_id in snr_ids:
        y_values.append([data[k][param][snr_id] for k in data])
    y_values = np.transpose(np.array(y_values))
    y_values = [mean(y_val_epoch) for y_val_epoch in y_values]
    plt.plot(x_epochs, y_values_train, '-ob', label='Train Loss')
    plt.plot(x_epochs, y_values, '-sr', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.title('Validation Loss values per epoch -- %s - %s'% (SNRS[snr_ids[0]], SNRS[snr_ids[-1]]))
    plt.legend()
    plt.savefig(pjoin(plot_fig_dir, '%s_%s_epochs_%d-%d_avg.png' % (SNRS[snr_ids[0]], SNRS[snr_ids[-1]], epochs[0], epochs[1])))
    plt.close()


def plot_test_MSE_losses_per_snr_ADMM_FISTANet(epoch, snr_ids, model_name):
    f = pjoin(test_loss_path, 'MSE_per_SNR_FISTA-Net_ADMM_imgs_10000_epoch_{}.ckpt'.format(epoch))
    checkpoint = torch.load(f)
    losses_ADMM = checkpoint['avg_test_losses_per_snr_ADMM']
    losses_FISTANet = checkpoint['avg_test_losses_per_snr_FISTANet']
    x_snrs = [int(SNRS[id][4:]) for id in snr_ids]
    plot_fig_dir = pjoin(plot_dir, model_name)
    plot_fig_dir = pjoin(plot_fig_dir, 'test_losses/')
    if not os.path.exists(plot_fig_dir):
        os.makedirs(plot_fig_dir)
    plt.figure()
    plt.plot(x_snrs, [losses_ADMM[i] for i in snr_ids], '-sr', label='ADMM')
    plt.plot(x_snrs, [losses_FISTANet[i] for i in snr_ids], '-ob', label='FISTA-Net')
    plt.xlabel('SNR')
    plt.ylabel('Test Loss (MSE)')
    #plt.title('Test MSE Loss values per SNR - epoch %d'% epoch)
    plt.legend()
    plt.savefig(pjoin(plot_fig_dir, 'MSE_per_SNR_FISTA-Net_ADMM_epoch-%d_%s-%s.png' % (epoch, SNRS[snr_ids[0]], SNRS[snr_ids[-1]])))
    plt.close()


def plot_test_MSE_losses_per_snr_ADMM_FISTANet_curvelet(epoch, snr_ids, model_name):
    f = pjoin(test_loss_path, 'MSE_per_SNR_FISTA-Net_ADMM_imgs_10000_epoch_{}.ckpt'.format(epoch))
    checkpoint = torch.load(f)
    losses_ADMM = checkpoint['avg_test_losses_per_snr_ADMM']
    losses_FISTANet = checkpoint['avg_test_losses_per_snr_FISTANet']
    x_snrs = [int(SNRS[id][4:]) for id in snr_ids]
    plot_fig_dir = pjoin(plot_dir, model_name)
    plot_fig_dir = pjoin(plot_fig_dir, 'test_losses/')
    mat_file = loadmat(curvelet_path)
    y_curvelet = np.transpose(np.array(mat_file['all_mse']))
    if not os.path.exists(plot_fig_dir):
        os.makedirs(plot_fig_dir)
    plt.figure()
    plt.plot(x_snrs, [x/(1000*255) for x in y_curvelet], '-vg', label='ADMM + Curvelet')
    plt.plot(x_snrs, [losses_ADMM[i]/(1000*255) for i in snr_ids], '-vr', label='ADMM + Abel')
    plt.plot(x_snrs, [losses_FISTANet[i]/(1000*255) for i in snr_ids], '-vb', label='FISTA-Net')
    plt.xlabel('SNR')
    plt.ylabel('MSE')
    #plt.title('Test MSE Loss values per SNR - epoch %d'% epoch)
    plt.legend()
    plt.savefig(pjoin(plot_fig_dir, 'VW_MSE_per_SNR_FISTA-Net_ADMM_curvelet_epoch-%d_%s-%s.png' % (epoch, SNRS[snr_ids[0]], SNRS[snr_ids[-1]])))
    plt.close()


if __name__ == '__main__':
    #plot_test_MSE_losses_per_snr_ADMM_FISTANet(31, list(range(10)), 'testing')
    plot_test_MSE_losses_per_snr_ADMM_FISTANet_curvelet(31, list(range(10)), 'testing')
    plot_param_per_epoch(epochs, 'train_losses', 'testing')
    plot_avg_val_loss_per_epoch_per_snr(epochs, 'testing')
    plot_avg_val_loss_per_snr_per_epoch(epochs, list(range(10)), 'testing')
    tests = [
        [list(range(10)), True],
        [[0, 3, 6, 9], True],
        [[6, 7, 8, 9], False],
        [[2, 4, 6, 9], False]]
    for _, (snr_ids, limit_axis) in enumerate(tests):
        plot_avg_val_loss_per_epoch(epochs, snr_ids, 'testing', limit_axis=limit_axis)
    plot_avg_val_loss_per_epoch_avg(epochs, list(range(10)), 'testing', limit_axis=False)
    #epochss = [(1, 12), (1, 14), (1, 16), (1, 21), (1, 25), (1, 30), (1, 32)]
    #for epochs in epochss:
    plot_train_and_avg_val_loss_per_epoch(epochs, list(range(10)), 'testing', limit_axis=False)
