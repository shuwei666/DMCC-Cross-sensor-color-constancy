"""
 Training
 If you use this code, please cite the following paper:
 Shuwei Yue and Minchen Wei. "Effective cross-sensor color constancy using a dual-mapping strategy". In JOSA A, 2023.

"""
__author__ = "Shuwei Yue"
__credits__ = ["Shuwei Yue"]

import argparse
import os
import torch
import numpy as np
import logging
import pprint
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import optim

from model import Dmcc
from dataset import CcData
from utils import loss_angular, set_seed, \
    LossTracker, save_log, Evaluator

try:
    from torch.utils.tensorboard import SummaryWriter

    USE_TB = False

except ImportError:

    USE_TB = False


def train_net(net,
              device,
              epochs=2000,
              batch_size=32,
              lr=7e-3,
              fold=0,
              validationFrequency=4,
              dir_data='',
              test_sensor='',
              stopping_patience=50,
              ):
    train_loader, val_loader = get_data_loaders(batch_size_train=batch_size, batch_size_eval=1,
                                                fold=fold, data_dir=dir_data, test_name=test_sensor)
    if USE_TB:
        writer = SummaryWriter(comment=f'_LR_{lr}_BS_{batch_size}')

    logging.info(f'''Start Training:
        Epochs:                 {epochs} epochs
        Batch size:             {batch_size}
        Learning rate           {lr}
        Validation Freq:        {validationFrequency}
        Device:                 {device}
        TensorBoard:            {USE_TB}   
        Train/Val dir:          {dir_data}
        Test sensor:            {test_sensor}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = loss_angular

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    evaluator = Evaluator()
    best_val_error, best_metrics = 100.0, evaluator.get_best_metrics()
    train_loss = LossTracker()

    epochs_without_improvement = 0

    for epoch in range(epochs):
        net.train()
        train_loss.reset()
        for batch in train_loader:
            img, illu_gd = batch
            img, illu_gd = img.to(device), illu_gd.to(device)

            optimizer.zero_grad()
            illu_pred, l1_loss = net(img)
            loss = criterion(illu_pred, illu_gd) + l1_loss
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item())

        if (epoch + 1) % validationFrequency == 0:

            evaluator.reset_errors()
            val_loss = val_net(net, criterion, val_loader, evaluator, device)

            if USE_TB:
                writer.add_scalars('Loss',
                                   {'train': train_loss.avg,
                                    'eval': val_loss.avg}, epoch)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

            if val_loss.avg < best_val_error:

                epochs_without_improvement = 0

                best_val_error = val_loss.avg
                best_metrics = evaluator.compute_metrics()
                logging.info(f'\n Best metrics:\n*************************'
                             f'\n {pprint.pformat(best_metrics)}'
                             f'\n ************************* \n')

                logging.info('Saving the best model...')
                if not os.path.exists('new_models'):
                    os.mkdir('new_models')
                if not os.path.exists('debug_models'):
                    os.mkdir('debug_models')
                torch.save(net.state_dict(), f"debug_models/" + f"dmcc__{dir_data.split('/')[-2]}"
                                                                             f"_test_{test_sensor}_debug.pth")
            else:
                epochs_without_improvement += 1

            logging.info(
                f'Epoch:[{epoch + 1}/{epochs}]=======>Train-mean:'
                f' {train_loss.avg:.2f}/{best_val_error:.2f}(Best_val-mean)')

        if validationFrequency * epochs_without_improvement >= stopping_patience:
            logging.info(f'Training stopped! Best model has been saved!')
            logging.info(f'\n Best metrics:\n*************************'
                         f'\n {pprint.pformat(best_metrics)}'
                         f'\n ************************* \n')

            break
        scheduler.step()
    if USE_TB:
        writer.close()

    logging.info('\n **************************************************\n'
                 '**************************************************\n'
                 '**************************************************\n')
    return best_metrics


def get_data_loaders(batch_size_train=32, batch_size_eval=1, fold=0, data_dir='', test_name=''):
    data_train = CcData(data_dir, train=True, fold_num=fold, test_sensor=test_name)
    train_loader = DataLoader(data_train, batch_size=batch_size_train, shuffle=True,
                              num_workers=8, drop_last=False, pin_memory=True)
    data_val = CcData(data_dir, train=False, fold_num=fold, test_sensor=test_name)
    val_loader = DataLoader(data_val, batch_size=batch_size_eval, shuffle=False,
                            num_workers=8, drop_last=False, pin_memory=True)

    logging.info(f'The training/val dataset is: {len(data_train)}/{len(data_val)}')

    return train_loader, val_loader


def val_net(net, criterion, eval_loader, evaluator, device):
    net.eval()
    val_loss = LossTracker()
    with tqdm(total=len(eval_loader), desc='Validation Process',
              bar_format='{l_bar}{bar:40}{r_bar}', colour='#00FF00') as pbar:
        for batch in eval_loader:
            img, illu_gd = batch
            img, illu_gd = img.to(device), illu_gd.to(device)

            with torch.no_grad():
                illu_pred, _ = net(img)
                loss = criterion(illu_pred, illu_gd)
                evaluator.add_error(np.round(loss.item(), 2))
                val_loss.update(loss.item())
                pbar.update()
        pbar.set_postfix(**{'Val-loss (epoch)': val_loss.avg})

    net.train()

    return val_loss


def get_args():
    parser = argparse.ArgumentParser(description='Training DMCC net.')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=10000,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', type=int, nargs='?', default=32,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-lr', '--learning-rate', type=float, nargs='?', metavar='LR', default=7e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('-d', '--fold', type=int, metavar='FOLD', default=0,
                        help='Testing fold, i.e., using fold0 as testing fold,'
                             ' which means the fold1 & fold2 are training folds.'
                        )
    parser.add_argument('-trd', '--training_dir', dest='trdir', type=str, default='./dataset/TAU_S_IMX135/',
                        help='Training dataset directory')
    # Calibrated matrix chosen from [Cubeplus, fuji, canon1ds, canon600d,
    # nikond5200, olympus, panasonic, samsung, sony, test_C_5DSR, test_S_IMX135, test_N_D810 ]
    parser.add_argument('-ts', '--testing_sensor', dest='test_sensor', type=str, default='Cubeplus',
                        help='Testing sensor, should be aligned with the calibrated diagonal matrix')

    parser.add_argument('-vafreq', '--validation-frequency', type=int, dest='val_freq', default=5,
                        help='Validation frequency')
    parser.add_argument('-seed', '--random-seed', type=int, metavar='SEED', dest='seed', default=666,
                        help='Random seed number for reproduction')
    parser.add_argument('-patience', '--early-stop-patience', type=int, metavar='P', dest='patience', default=800,
                        help='The patience number of epochs for early stopping')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logging.info('Training of cross-sensor color constancy: DMCC')

    args = get_args()
    save_log(log_dir, data_name=args.trdir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    logging.info(f'Using seed {args.seed}')

    net = Dmcc()

    net.to(device)
    train_net(net=net,
              epochs=args.epochs,
              lr=args.lr,
              batch_size=args.batchsize,
              fold=args.fold,
              device=device,
              validationFrequency=args.val_freq,
              dir_data=args.trdir,
              test_sensor=args.test_sensor,
              stopping_patience=args.patience)
