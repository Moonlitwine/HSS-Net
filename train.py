import argparse
import shutil
import logging
import yaml
import sys
import os
from datetime import datetime # 导入 datetime 模块
from core.configs.init_configs import get_cfg_defaults
from core.models.train_models import Restoration
import torch
import numpy as np
from core.dataset import dataloader
import torch.utils.tensorboard as tb

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument("--cfg", type=str, required=True, help="Name of the config file")
    parser.add_argument("--comment", type=str, default="", help="A string for experiment comment")
    parser.add_argument("--exp", type=str, default="exp", help="Path for saving running related data")
    parser.add_argument("--doc",
        type=str,
        default="doc",
        help="A string for documentation purpose. "
             "Will be the name of the log folder.")
    parser.add_argument("--resume", type=str, default="", help="Path to the checkpoint file")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--val_folder",
        type=str,
        default="val_image",
        help="Folder name of verification results")
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="The device that runs the model")
    parser.add_argument(
        '--parallel', action='store_true', default=False, help="Whether to use the DataParallel computing mode.")
    parser.add_argument('--no_patch', action='store_true', default=False, help="No patching method used during testing")
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file("configs/" + args.cfg)
    cfg.comment = args.comment
    cfg.exp = args.exp
    cfg.doc = args.doc
    cfg.resume = args.resume
    cfg.seed = args.seed
    cfg.val_folder = args.val_folder

    
    if cfg.resume == "":
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
       
        cfg.exp = f"{cfg.exp}_{timestamp}"
        print(f"INFO: Starting a NEW experiment. Output directory: {cfg.exp}")
    else:
       
        print(f"INFO: Resuming experiment from checkpoint. Output directory (used for logs/tensorboard): {cfg.exp}")


    log_path = os.path.join(cfg.exp, "logs", cfg.doc)
    tb_path = os.path.join(cfg.exp, "tensorboard")
    val_folder = os.path.join(cfg.exp, cfg.val_folder)
    cfg.val_folder = val_folder 

    os.makedirs(cfg.exp, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(tb_path, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)


    with open(os.path.join(cfg.exp, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    cfg.data.model_param_dir = cfg.exp

    args.tb_loss = tb.SummaryWriter(log_dir=os.path.join(tb_path, "loss"))
    args.tb_pixell1weight = tb.SummaryWriter(log_dir=os.path.join(tb_path, "pixell1weight"))

    level = getattr(logging, cfg.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(cfg.verbose))
    handler1 = logging.FileHandler(os.path.join(log_path, "stdout.txt"))
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    if args.device == "":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        logging.info("Using device: {}".format(device))
        args.device = device
    else:
        args.device = torch.device(args.device)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = True

    cfg.freeze()
    return args, cfg

def main():
    args, config = parse_args_and_config()
    log_path = os.path.join(config.exp, "logs", config.doc) 
    logging.info("Writing log file to {}".format(log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(config.comment))
    # data loading
    print("=> using dataset '{}'".format(config.data.dataset))
    DATASET = dataloader.DataLoader(config)
    # create model
    print("=> creating restoration model...")
    restorer = Restoration(args, config)
    restorer.train(DATASET)
    return 0

if __name__ == "__main__":
    sys.exit(main())