import os
import time
import logging
from datetime import datetime
from collections import defaultdict
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from evaluate.evaluate import evaluate, evaluate_classic
from model.model import load_FCR
from dataset.dataset import load_dataset_splits
from utils.general_utils import initialize_logger, ljson
from utils.data_utils import data_collate
import argparse
from train import train
import json

def parse_arguments():
    """
    Read arguments when this script is called from a terminal.
    Can either specify a config file or introduce arguments from terminal.
    Help guide available in the latetr case
    """

    # interaction mode
    parser_init = argparse.ArgumentParser()
    parser_init.add_argument("--file", type=str, default="console", help="Path to config file. If not provided, CLI args are used")
    args, remaining_args = parser_init.parse_known_args()

    if args.file != "console":
        with open(args.file) as f:
            return json.load(f)
    else:
        parser = argparse.ArgumentParser(
            parents=[parser_init],
            description = "Experiment arguments"
        )
        # setting arguments
        parser.add_argument('--name', default='default_run')
        parser.add_argument("--artifact_path", type=str, required=True)
        parser.add_argument("--data_path", type=str, required=True)
        parser.add_argument("--graph_path", type=str, default=None) 
        parser.add_argument('--cpu', action='store_true')
        parser.add_argument('--gpu', default='0')
    
        # model arguments
        parser.add_argument("--omega0", type=float, default=1.0, help="weight for reconstruction loss")
        parser.add_argument("--omega1", type=float, default=10.0, help="weight for distribution loss")
        parser.add_argument("--omega2", type=float, default=0.1, help="weight for KL divergence")
        parser.add_argument("--dist_outcomes", type=str, default="normal", help="nb;zinb;normal")
        parser.add_argument("--dist_mode", type=str, default="match", help="classify;discriminate;fit;match")
        parser.add_argument("--distance", type=str, default="element")
    
        # used keys arguments (added)
        parser.add_argument("--covariate_keys")
        parser.add_argument("--perturbation_key")
        parser.add_argument("--split")
        parser.add_argument("--embed_outcomes", type=str, default=True)
        parser.add_argument("--embed_treatments", type=str, default=False)
        parser.add_argument("--embed_covariates", type=str, default=True)
        
        # hparams
        parser.add_argument("--hparams", type=str, default="")
    
        # training arguments
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--max_epochs", type=int, default=2000)
        parser.add_argument("--patience", type=int, default=20)
        parser.add_argument("--checkpoint_freq", type=int, default=10)
        parser.add_argument("--eval_mode", type=str, default="native", help="classic;native")
        parser.add_argument("--adv_epoch", type=int, default=5, help="number of epochs between adversarial training")

        return dict(vars(parser.parse_args()))

# Initialize workflow
train(parse_arguments())