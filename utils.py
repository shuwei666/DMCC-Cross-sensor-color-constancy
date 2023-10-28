import matplotlib.pyplot as plt
from torch import Tensor
import numpy as np
import pandas as pd
import fnmatch
import os
import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
import random
import torch
import math
from torch.nn.functional import normalize
import yaml
import logging
from sklearn.cluster import KMeans
import glob

EPS = 1e-10


def norm_img(img):
    """ normalized image values to [0., 1.]"""
    img = img / (img.max() + EPS)

    img = img.clip(0., 1.)
    return img


def hwc_to_chw(img: np.ndarray):
    """ Converts an image from height * width * channel to (channel * height * width)"""
    return img.transpose(2, 0, 1)


def chw_to_hwx(x: Tensor) -> Tensor:
    """ Converts a Tensor to an Image """
    img = x.cpu().numpy()
    img = img.transpose(0, 2, 3, 1)[0, :, :, :]
    return img


def print_metrics(current_metrics: dict, best_metrics: dict):
    print(" Mean ......... : {:.4f} (Best: {:.4f})".format(current_metrics["mean"], best_metrics["mean"]))
    print(" Median ....... : {:.4f} (Best: {:.4f})".format(current_metrics["median"], best_metrics["median"]))
    print(" Trimean ...... : {:.4f} (Best: {:.4f})".format(current_metrics["trimean"], best_metrics["trimean"]))
    print(" Best 25% ..... : {:.4f} (Best: {:.4f})".format(current_metrics["bst25"], best_metrics["bst25"]))
    print(" Worst 25% .... : {:.4f} (Best: {:.4f})".format(current_metrics["wst25"], best_metrics["wst25"]))
    print(" Worst 5% ..... : {:.4f} (Best: {:.4f})".format(current_metrics["wst95"], best_metrics["wst95"]))
    print(" Best ......... : {:.4f} (Best: {:.4f})".format(current_metrics["bst"], best_metrics["bst"]))


def print_single_metric(current_metrics):
    print(" Mean ......... : {:.4f}".format(current_metrics["mean"]))
    print(" Median ....... : {:.4f}".format(current_metrics["median"]))
    print(" Trimean ...... : {:.4f}".format(current_metrics["trimean"]))
    print(" Best 25% ..... : {:.4f}".format(current_metrics["bst25"]))
    print(" Worst 25% .... : {:.4f}".format(current_metrics["wst25"]))
    print(" Worst 5% ..... : {:.4f}".format(current_metrics["wst95"]))
    print(" Best ......... : {:.4f}".format(current_metrics["bst"]))


def save_log(best_stats, current_loss, training_loss, val_loss, path_to_log):
    log_data = pd.DataFrame({
        'Tr-loss': [training_loss],
        'Val-loss': [val_loss],
        'b-mean': best_stats['mean'],
        'b-median': best_stats['median'],
        'b-tri-mean': best_stats['trimean'],
        'b-b25': best_stats['bst25'],
        'b-wst25': best_stats['wst25'],
        'b-wst': best_stats['wst95'],
        'b-bst': best_stats['bst'],
        **{k: [v] for k, v in current_loss.items()}
    })
    head = log_data.keys() if not os.path.exists(path_to_log) else False
    log_data.to_csv(path_to_log, mode='a', header=head, index=False)


def log_sys(args):
    dt = datetime.now()
    path_to_log = os.path.join('./log', args.data_name,
                               f'fold_{args.fold_num}_'
                               f'-{dt.day}-{dt.hour}-{dt.minute}')

    os.makedirs(path_to_log, exist_ok=True)
    path_to_metrics_log = os.path.join(path_to_log, 'error.csv')
    vis_log_tr = os.path.join(f'./vis_log', f'{dt.day}-{dt.hour}-{dt.minute}', 'train')
    vis_log_acc = os.path.join(f'./vis_log', f'{dt.day}-{dt.hour}-{dt.minute}', 'acc')
    os.makedirs(vis_log_tr, exist_ok=True)
    os.makedirs(vis_log_acc, exist_ok=True)

    param_info = {'lr': args.lr, 'batch_size': args.batch_size,
                  'fold_num': args.fold_num, 'data_name': args.data_name,
                  'time_file': f'{dt.day}-{dt.hour}-{dt.minute}',
                  'seed': f'{args.seed}'}

    return SummaryWriter(vis_log_tr), SummaryWriter(vis_log_acc), \
           path_to_log, path_to_metrics_log, param_info


def k_fold(n_splits=3, num=0):
    """
    Randomly Split the training and testing datasets.
    """
    assert n_splits is 3, "three-cross validation"
    num = np.arange(num)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=666)
    tr, te = [], []
    train_test = {}
    for train, test in kf.split(num):
        tr.append((train.tolist()))
        te.append((test.tolist()))

    train_test['train'] = tr
    train_test['test'] = te

    return train_test


class AwbAug:
    def __init__(self, illu_path, sensor_name):
        """
        Data Augmentation method, i.e., AWB-Aug.

        A series of illuminants are selected from all images in the datasets as centers of circles.
        Then the data augmentation was performed by randomly assigning the RGB values with the
        chromaticity distance to the selected illuminants smaller than 0.05,
        i.e., I_aug = I_origin * np.diag(illu_aug / illu_gd)
        """

        self.img = None
        self.illu = None

        self.illu_path = self.__load_illu(illu_path)
        self.sensor_name = sensor_name
        self.transform_illu_path()

    def __load_illu(self, illu_path):
        if not isinstance(illu_path,  list):
            raise TypeError('illu_path should be a list')
        return np.array([np.load(i) for i in illu_path])

    def transform_illu_path(self):
        martix_files = [f for f in os.listdir('./calibrated_diagonal_matrix/') if
                        fnmatch.fnmatch(f, f'*{self.sensor_name}*.npy')]
        if not martix_files:
            raise ValueError(f"No .npy file contains {self.sensor_name} found, please check your file name"
                             f"and the related file location")
        if len(martix_files) > 2:
            raise ValueError(f"There is only one matrix should be used!  check your matrix name.")

        M = np.load(os.path.join('./calibrated_diagonal_matrix', martix_files[0]))
        # logging.info(f"Calibrated diagonal matrix M is: {martix_files[0]}")
        after_diagonal_mapping_illu = np.dot(self.illu_path, M)
        self.illu_path = after_diagonal_mapping_illu[np.all(after_diagonal_mapping_illu > 0.0, axis=1)]
        self.illu_path = self.illu_path / self.illu_path.sum(axis=1).reshape(-1, 1)

    def __circle_point(self, illu, radius=0.05):
        while True:
            res_r = random.uniform(illu[0] - radius, illu[0] + radius)
            res_g = random.uniform(illu[1] - radius, illu[1] + radius)
            dis = (res_r - illu[0]) ** 2 + (res_g - illu[1]) ** 2
            if dis <= radius ** 2 and (res_r + res_g) < 0.9999 and res_r > 0 and res_g > 0:
                return np.array([res_r, res_g, 1 - res_r - res_g])

    def awb_aug(self, gd, img):
        self.illu = gd
        self.img = img
        random_illu = self.illu_path[np.random.choice(self.illu_path.shape[0]), :]
        aug_illu = self.__circle_point(random_illu)
        new_img = np.dot(self.img, np.diag(aug_illu / self.illu))

        return norm_img(new_img), aug_illu


def loss_angular(pred, label, safe_v=0.999999):
    chrom_b = (1 - pred.sum(axis=1)).unsqueeze(1)
    pred = torch.cat((pred, chrom_b), dim=1)

    dot = torch.clamp(torch.sum(normalize(pred, dim=1) * normalize(label, dim=1), dim=1), -safe_v, safe_v)
    angle = torch.acos(dot) * (180 / math.pi)

    return torch.mean(angle)


def loss_angular_eval(pred, label, safe_v=0.999999):
    dot = torch.clamp(torch.sum(normalize(pred, dim=1) * normalize(label, dim=1), dim=1), -safe_v, safe_v)
    angle = torch.acos(dot) * (180 / math.pi)

    return torch.mean(angle)


class AngularError:
    def __init__(self, eval_mode=False, k_center=2, data_path=''):
        super().__init__()
        self.eval = eval_mode
        self.centers = k_center
        self.data_dir = data_path

    def get_illu(self):
        km = KMeans(n_clusters=self.centers, random_state=666)
        illu_list = glob.glob(self.data_dir + 'numpy_labels' + '/*.npy')
        illu_all = []
        for i in illu_list:
            illu_data = np.load(i)
            illu_all.append(illu_data)
        illu_all = np.array(illu_all)
        km.fit(illu_all)
        centers = km.cluster_centers_

        return centers

    def center(self, gd):
        # calc the k_center
        centers = self.get_illu()
        sum_dist = ((gd - centers) ** 2).sum(axis=1)
        gd_center = centers[torch.argmin(sum_dist), :]

        return gd_center.reshape(1, -1)

    def compute(self, pred, label, safe_v=0.999999):
        # add one col
        add_c = (1 - pred.sum(axis=1)).unsqueeze(1)
        pred = torch.cat((pred, add_c), dim=1)
        if label.shape[1] is not 3:
            add_c = (1 - label.sum(axis=1)).unsqueeze(1)
            label = torch.cat((label, add_c), dim=1)

        if self.eval:
            label = self.center(label)

        dot = torch.clamp(torch.sum(normalize(pred, dim=1) * normalize(label, dim=1), dim=1), -safe_v, safe_v)
        angle = torch.acos(dot) * (180 / math.pi)

        # return torch.max(angle).to(DEVICE) # USING THE MAX-ERROR!
        return torch.mean(angle)


def set_seed(seed=666):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False


class LossTracker:
    """Tracker Loss and do the calculation"""

    def __init__(self):
        self.val, self.avg, self.sum, self.count, self.max = 0, 0, 0, 0, 0
        self.loss = []

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.loss.append(self.val)
        self.count += n
        self.sum += self.val * n
        self.avg = self.sum / self.count

        self.max = np.max(np.array(self.loss))

    def get_loss(self):
        return self.avg


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def feature_select(img_tmp, thresh_dark=0.00, thresh_saturation=0.98):
    """
    The four feature selected, i.e., bright, max, mean and dark pixels
    """
    img_debug = img_tmp.reshape(-1, 3)

    img_tmp = img_debug.copy()
    img_tmp = img_tmp[np.all(img_tmp > thresh_dark, axis=1), :]
    img_tmp = img_tmp[np.all(img_tmp < thresh_saturation, axis=1), :]
    # 0. Brightest pixel
    bright_v = img_tmp[np.argmax(img_tmp.sum(axis=1))]
    # 1. Maximum pixel
    max_wp = img_tmp.max(axis=0)
    # 2. Average pixel
    mean_v = img_tmp.mean(axis=0)
    # 3. Darkest pixel
    dark_v = img_tmp[np.argmin(img_tmp.sum(axis=1))]

    feature_data = np.vstack([bright_v, max_wp, mean_v, dark_v])
    feature_data /= (feature_data.sum(axis=1).reshape(-1, 1) + EPS)
    feature_data = feature_data[:, :2]

    return feature_data


def save_config(config, log_file):
    with open(log_file, 'a') as f:
        for section, options in config.items():
            f.write('[{}]\n'.format(section))
            for key, value in options.items():
                f.write('{}: {}\n'.format(key, value))
        f.write('\n')


def save_log(log_dir, data_name):
    log_filename = datetime.datetime.now().strftime("%H-%M-") + f"{data_name.split('/')[-2]}" + ".txt"
    log_file = os.path.join(log_dir, log_filename)
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger('').addHandler(file_handler)


class Evaluator:

    def __init__(self):
        monitored_metrics = ["mean", "median", "trimean", "bst25", "wst25", "wst95", 'bst']
        self.__errors = {}
        self.__metrics = {}
        self.__best_metrics = {m: 100.0 for m in monitored_metrics}

    def get_best_metrics(self):
        return self.__best_metrics

    def add_error(self, error):
        self.__errors.append(error)
        return self

    def reset_errors(self):
        self.__errors = []

    def get_errors(self):
        return self.__errors

    def compute_metrics(self):
        self.__errors = sorted(self.__errors)
        # self.__errors = self.__errors[~np.isnan(self.__errors)]
        self.__metrics = {
            'mean': np.mean(self.__errors),
            'median': np.median(self.__errors),
            'trimean': 0.25 * self.__g(0.25) + 0.5 * self.__g(0.5) + 0.25 * self.__g(0.75),
            'bst25': np.mean(self.__errors[:int(len(self.__errors) * 0.25)]),
            'wst25': np.mean(self.__errors[int(len(self.__errors) * 0.75):]),
            # 'wst95': np.mean(self.__errors[int(len(self.__errors) * 0.95):]),
            # 'bst': np.min(self.__errors)
        }

        return self.__metrics

    def update_best_metrics(self) -> dict:
        self.__best_metrics["mean"] = self.__metrics["mean"]
        self.__best_metrics["median"] = self.__metrics["median"]
        self.__best_metrics["trimean"] = self.__metrics["trimean"]
        self.__best_metrics["bst25"] = self.__metrics["bst25"]
        self.__best_metrics["wst25"] = self.__metrics["wst25"]
        self.__best_metrics["wst95"] = self.__metrics["wst95"]
        self.__best_metrics["bst"] = self.__metrics["bst"]

        return self.__best_metrics

    def __g(self, f):
        return np.percentile(self.__errors, f * 100)


def param_calc(net):
    total_params = 0
    for param in net.parameters():
        total_params += param.numel()

    return total_params * 4 / 1024 / 1024


def img_rg(img, thres=0.0000001):
    "calc img rg and remove dark pixels"
    img = img.reshape(-1, 3)
    img_new = img[np.all(img > thres, axis=1), :]
    img_rgb = img_new / img_new.sum(axis=1).reshape(-1, 1)
    return img_rgb


def tau_ccts():
    """
    from A: 2800, F11：4000, to D65：6500
    :return:
    """
    camera_cct = {
        'S_IMX135': [[0.41, 0.42, 0.17],
                     [0.31, 0.47, 0.23],
                     [0.23, 0.45, 0.32]],
        'C_5DSR': [[0.35, 0.48, 0.17],
                   [0.27, 0.50, 0.22],
                   [0.19, 0.49, 0.32]],
        'N_D810': [[0.41, 0.45, 0.14],
                   [0.22, 0.46, 0.33],
                   [0.21, 0.44, 0.35]],
    }

    return camera_cct


def plot_results(vis_gd, vis_pred, fold):
    """for vis of test results"""
    vis_gd = np.array(vis_gd) if not isinstance(vis_gd, np.ndarray) else vis_gd
    vis_pred = np.array(vis_pred) if not isinstance(vis_pred, np.ndarray) else vis_pred

    plt.scatter(vis_gd[:, 0], vis_gd[:, 1], c='r', marker='p', label='gt')
    plt.scatter(vis_pred[:, 0], vis_pred[:, 1], c='g', marker='.', label='pred')
    plt.xlim([0.1, 0.7])
    plt.ylim([0.3, 0.7])
    plt.legend(title=f'Fold {fold}')
    plt.show()
