import numpy as np
import glob
import random

from torch.utils.data import Dataset
from utils import k_fold, AwbAug, feature_select


class CcData(Dataset):
    def __init__(self, path, train=True, fold_num=0, test_sensor=''):
        self.path = path
        self.train = train
        self.illu_full = glob.glob(path + 'numpy_labels' + '/*.npy')
        self.img_full = glob.glob(path + 'numpy_data' + '/*.npy')
        self.img_full.sort(key=lambda x: x.split('\\')[-1].split('_')[-1].split('.')[0])
        self.illu_full.sort(key=lambda x: x.split('\\')[-1].split('_')[-1].split('.')[0])

        train_test = k_fold(n_splits=3, num=len(self.img_full))
        img_idx = train_test['train' if self.train else 'test'][fold_num]

        self.fold_data = [self.img_full[i] for i in img_idx]
        self.fold_illu = [self.fold_data[i].replace('numpy_data', 'numpy_labels') for
                          i in range(len(self.fold_data))]
        self.data_aug = AwbAug(self.illu_full, sensor_name=test_sensor)

    def __len__(self):
        return len(self.fold_data)

    def __getitem__(self, idx):
        """ Gets next data in the dataloader.

        Note: We pre-processed the input data in the format of '.npy' for fast processing. If
        you want to train your own dataset, the corresponding of loadig image should also be changed.

        """
        img_data = np.load(self.fold_data[idx])
        gd_data = np.load(self.fold_illu[idx])
        # if self.train:
        img_data, gd_data = self.data_aug.awb_aug(gd_data, img_data)
        feature_data = feature_select(img_data)

        return feature_data.astype(np.float32), gd_data.astype(np.float32)


class CcDataEval(Dataset):
    """
    for evaluation
    """

    def __init__(self, path, train=False, fold_num=0):
        self.path = path
        self.train = train

        self.img_full = glob.glob(path + 'numpy_data' + '/*.npy')
        self.img_full.sort(key=lambda x: x.split('\\')[-1].split('_')[-1].split('.')[0])

        train_test = k_fold(n_splits=3, num=len(self.img_full))
        img_idx = train_test['train' if self.train else 'test'][fold_num]

        self.fold_data = [self.img_full[i] for i in img_idx]

        self.fold_illu = [self.fold_data[i].replace('numpy_data', 'numpy_labels') for
                          i in range(len(self.fold_data))]

    def __len__(self):
        return len(self.fold_data)

    def random_select(self, num=5):
        return random.sample(self.fold_data, num)

    def __getitem__(self, idx):
        """ Gets next data in the dataloader.

        Note: We pre-processed the input data in the format of '.npy' for fast processing. If
        you want to train your own dataset, the corresponding of loadig image should also be changed.

        """
        img_data = np.load(self.fold_data[idx])
        gd_data = np.load(self.fold_illu[idx])
        gd_data = gd_data / gd_data.sum()
        img_name = self.fold_illu[idx].split('.npy')[0].split('/')[-1]
        feature_data = feature_select(img_data)
        return feature_data.astype(np.float32), gd_data.astype(np.float32), img_name
