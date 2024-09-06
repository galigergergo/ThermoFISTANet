import torch
from os.path import dirname, join as pjoin
from statistics import mean

save_path = './models/testing/'

epochs = 9

for i in range(1, epochs):
    f = pjoin(save_path, 'epoch_{}.ckpt'.format(i))
    checkpoint = torch.load(f)
    print(i)
    print(checkpoint['optimizer']['param_groups'][0]['lr'])
    print(mean(checkpoint['train_losses']))
    print(checkpoint['avg_val_losses_per_snr'])
    print()
