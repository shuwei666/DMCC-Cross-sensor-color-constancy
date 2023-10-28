import argparse
import pprint
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from model import Dmcc
from dataset import CcDataEval
from utils import loss_angular_eval, set_seed, plot_results, Evaluator

FOLDS = 3
MODEL_PATH = './pretrained-models/dmcc__TAU_S_IMX135_test_Cube+.pth'
TEST_DATA_PATH = './dataset/Cube+/'


def load_and_test_model(net, device, test_loader, fold,
                        fold_evaluator, all_evaluator, error_name):

    fold_evaluator.reset_errors()
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    net.to(device)
    net.eval()
    vis_pred, vis_gd = [], []
    for batch in test_loader:
        img, illu_gd, img_name = batch
        img, illu_gd = img.to(device), illu_gd.to(device)

        illu_pred, _ = net(img)
        with torch.no_grad():
            chrom_b = (1 - illu_pred.sum(axis=1)).unsqueeze(1)
            illu_pred = torch.cat((illu_pred, chrom_b), dim=1)

            vis_pred.append(illu_pred.cpu().squeeze().numpy())
            vis_gd.append(illu_gd.cpu().squeeze().numpy())
            loss = loss_angular_eval(illu_pred, illu_gd)

            fold_evaluator.add_error(np.round(loss.item(), 2))
            all_evaluator.add_error(np.round(loss.item(), 2))

            error_name[str(img_name)] = np.round(loss.item(), 2)

    logging.info(f'fold {fold} error: \n'
                 f'{pprint.pformat(fold_evaluator.compute_metrics())}')

    plot_results(vis_gd, vis_pred, fold)


def get_args():
    parser = argparse.ArgumentParser(description='DMCC Testing Processing.')
    parser.add_argument('-d', '--fold', type=int, metavar='FOLD', default=0,
                        help='Testing fold, we also use three-fold to align with the area,'
                             'however, all the folds are testing folds.')
    parser.add_argument('-trd', '--training_dir', dest='trdir', type=str, default=TEST_DATA_PATH,
                        help='testing dataset directory'),
    parser.add_argument('-seed', '--random-seed', type=int, metavar='SEED', dest='seed', default=666,
                        help='Random seed number for reproduction')
    return parser.parse_args()


def get_data_loaders(batch_size_eval=1, fold=0, data_dir=''):
    data_val = CcDataEval(data_dir, train=False, fold_num=fold)
    val_loader = DataLoader(data_val, batch_size=batch_size_eval, shuffle=False,
                            num_workers=8, drop_last=False, pin_memory=True)

    logging.info(f'The Val dataset is:{data_dir} and length is: {len(data_val)}')

    return val_loader


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    set_seed(args.seed)
    net = Dmcc()

    data_name = Path(args.trdir).parts[-2]
    fold_evaluator, all_evaluator = Evaluator(), Evaluator()
    all_error = []
    error_name = {}
    all_evaluator.reset_errors()

    for fold in range(FOLDS):
        test_loader = get_data_loaders(batch_size_eval=1, fold=fold, data_dir=args.trdir)
        load_and_test_model(net, device, test_loader, fold, fold_evaluator, all_evaluator, error_name)

    all_error = all_evaluator.compute_metrics()
    df = pd.DataFrame(list(error_name.items()), columns=['Image_Name', 'Loss'])

    df.to_csv(f'test_{data_name}_error_name.csv', index=False)

    logging.info(f'Mean error of {data_name} is: \n\n:'
                 f'{pprint.pformat(all_error)}')


if __name__ == '__main__':
    main()
