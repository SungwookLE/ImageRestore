import os
import glob
import cv2
import matplotlib.pyplot as plt
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

from sklearn.model_selection import KFold
import albumentations as A
import tqdm

from multiprocessing import Pool

def cut_img(path):

    lr_path = path[0]
    hr_path = path[1]
    
    patch_size = 256
    stride = int(patch_size / 2)
    
    num = 0
    lr_img = cv2.imread(lr_path)[...,::-1]
    lr_img = cv2.resize(lr_img, dsize=(1024,1024), interpolation=cv2.INTER_CUBIC)

    hr_img = cv2.imread(hr_path)[...,::-1]

    print(lr_path)
    
    for top in range(0, lr_img.shape[0], stride):
        for left in range(0, lr_img.shape[1], stride):
            
            lr_name = lr_path.split('/')[-1].split('.')[0]
            hr_name = hr_path.split('/')[-1].split('.')[0]
            
            piece_lr = np.zeros([patch_size, patch_size, 3], np.uint8)
            temp_lr = lr_img[top : top+patch_size, left : left+patch_size, :]
            piece_lr[:temp_lr.shape[0], :temp_lr.shape[1], :] = temp_lr

            piece_hr = np.zeros([patch_size, patch_size, 3], np.uint8)
            temp_hr = hr_img[top : top+patch_size, left : left+patch_size, :]
            piece_hr[:temp_hr.shape[0], :temp_hr.shape[1], :] = temp_hr

            lr_name = lr_name + '_{}'.format(num)
            hr_name = hr_name + '_{}'.format(num)
            
            np.save('data/patch/lr/{}.npy'.format(lr_name), lr_img)
            np.save('data/patch/hr/{}.npy'.format(hr_name), hr_img)

            num+=1



if __name__ == "__main__":

    os.makedirs('data/patch/lr', exist_ok=True)
    os.makedirs('data/patch/hr', exist_ok=True)

    lr_path = np.array(sorted(glob.glob('data/train/lr/*.jpg')))[0:100]
    hr_path = np.array(sorted(glob.glob('data/train/hr/*.jpg')))[0:100]
    test_path = np.array(sorted(glob.glob('data/test/lr/*.jpg')))
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state = 357)
    for fold, (train_index, val_index) in enumerate(kf.split(lr_path)):
        lr_path_train, lr_path_val = lr_path[train_index], lr_path[val_index]
        hr_path_train, hr_path_val = hr_path[train_index], hr_path[val_index]
        break

    print(len(lr_path_train), len(lr_path_val))


    temp = list()
    for i in range(len(lr_path_train)):
        temp += [[lr_path_train[i], hr_path_train[i]]]

    with Pool(3) as p:
        p.map(cut_img, temp)