import argparse
import logging
import os
import re

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import yaml

from core.dataset import dataloader
from core.configs.init_configs import get_cfg_defaults
from core.models.restoration import DoRestoring
from core.models.train_models import Restoration


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument("--cfg", type=str, required=True, help="Path to the config file")
    parser.add_argument("--resume", type=str, default='', help="Path to the checkpoint file")
    parser.add_argument("--sample_set", type=str, default='../dataset/SPA+', help="Path to the test set")
    parser.add_argument(
        "--sample_folder", default='exp/sample_result', type=str, help="Location to save restored images")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--device", type=str, default="", help="The device that runs the model")
    parser.add_argument(
        '--parallel', action='store_true', default=False, help="Whether to use the DataParallel computing mode.")
    parser.add_argument(
        '--not_save_result', action='store_true', default=False, help="Do not save results")
    parser.add_argument('--calc_in_Y', action='store_true', default=False, help="Calculate PSNR and SSIM in Y channel")
    parser.add_argument('--no_patch', action='store_true', default=False, help="No patching method used during testing")

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    cfg.merge_from_file("configs/" + args.cfg)
    cfg.resume = args.resume
    cfg.data.dataset_dir = args.sample_set
    cfg.data.dataset = re.split('/', args.sample_set)[-1]

    cfg.sample_folder = args.sample_folder
    cfg.seed = args.seed

    if not os.path.exists(cfg.sample_folder):
        os.makedirs(cfg.sample_folder)

    # save current experimental configuration
    with open(os.path.join(cfg.sample_folder, cfg.data.dataset + "_config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # add device
    if args.device == "":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        logging.info("Using device: {}".format(device))
        args.device = device
    else:
        args.device = torch.device(args.device)

    # set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    torch.backends.cudnn.benchmark = True

    # freeze configuration parameters
    cfg.freeze()

    return args, cfg


def main():
    args, config = parse_args_and_config()

    # data loading
    print("=> using dataset '{}'".format(config.data.dataset))
    DATASET = dataloader.DataLoader(config)
    val_loader = DATASET.get_loaders(is_train=False)

    # create model
    print("=> creating Restoration model")
    restorer = Restoration(args, config)
    model = DoRestoring(restorer, args, config)
    model.restore(val_loader)

    return 0


if __name__ == '__main__':
    main()
