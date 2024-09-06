import cv2
from scipy.io import loadmat
from os.path import join as pjoin

data_dir = './test_vws'

x_imgs = []
preds = []

for snr_id in range(10):
    fle = pjoin(data_dir, 'SNR_%d_ADMM_FISTA-Net.mat' % snr_id)
    mat_file = loadmat(fle)
    x_imgs.append(mat_file['admm'])
    preds.append(mat_file['fistanet'])

print(x_imgs)



