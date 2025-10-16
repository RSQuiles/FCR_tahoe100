import os
import time
import logging
from datetime import datetime
from collections import defaultdict
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from .evaluate.evaluate import evaluate, evaluate_classic
from .model.model import load_FCR
from .dataset.dataset import load_dataset_splits
from .utils.general_utils import initialize_logger, ljson
from .utils.data_utils import data_collate
import argparse
from .train import train
from .train import prepare
import json
import argparse

"""
Fetch the latest model (path) given the directory where checkpoints are saved
We can also specify a given epoch
"""
def fetch_latest(directory, suffix=".pt", target_epoch=None):
    if target_epoch is not None:
        saves_dir = os.path.join(directory,"saves")
        if not os.path.isdir(saves_dir):
            raise FileNotFoundError(f"Directory not found: {saves_dir}")

        # Retrieve all model checkpoints in the saves directory
        ckpts = []    
        for root, _, files in os.walk(saves_dir):
            for f in files:
                if f.endswith(suffix):
                    ckpts.append(os.path.join(root, f))
        # Get the chekpoint corresponding to the specified epoch
        target_file = next((f for f in ckpts if f"epoch={target_epoch}.pt" in f), None)
        if target_file is None:
            raise FileNotFoundError(f"No checkpoint file found for epoch {target_epoch} in {saves_dir}")

        return target_file

    # If no epoch is specified, return the latest checkpoint
    saves_dir = os.path.join(directory,"saves")
    if not os.path.isdir(saves_dir):
        raise FileNotFoundError(f"Directory not found: {saves_dir}")

    # Retrieve all model checkpoints in the saves directory
    ckpts = []    
    for root, _, files in os.walk(saves_dir):
        for f in files:
            if f.endswith(suffix):
                ckpts.append(os.path.join(root, f))

    if not ckpts:
        raise FileNotFoundError(f"No checkpoint files with suffix '{suffix}' found in {saves_dir}")
    
    # Get the latest checkpoint
    latest_ckpt = max(ckpts, key=os.path.getmtime)
    return latest_ckpt # returns the string path of the latest checkpoint file

def get_model(model_dir, target_epoch=None, dataset="all"):
    # Get path to model checkpoint
    model_path = fetch_latest(model_dir, target_epoch=target_epoch)

    # LOAD ARGUMENTS
    # Note: dataset_mode defines which dataset split will be in the loader
    fcr_model = FCR_sim(model_path=model_path, dataset_mode=dataset)
    args = fcr_model.arguments

    # LOAD MODEL AND DATASETS
    trained_model = fcr_model.model
    datasets = fcr_model.dataset

    return [args, trained_model, datasets]


"""
This class allows to: 
- Start a training given a config file
- Import a model given the path to the checkpoint file
"""
class FCR_sim:
    def __init__(self, config_path=None, model_path=None, dataset_mode="train", parameters=None):
        # Importing mode for dataset
        self.dataset_mode = dataset_mode
        if model_path != None:
            self.load_model(model_path, self.dataset_mode)
        elif config_path != None:
            self.arguments = self.parse_arguments(config_path, parameters)
        else:
            raise ValueError("Valid config file or model path required")

    def parse_arguments(self, path, input_args):
        """
        Read arguments from config file
        """
        with open(path, "r") as f:
            arguments = json.load(f)
        """
        Arguments can also be passed when calling the sbatch job,
        in which case the values are modified
        """
        if input_args != None:
            hparams = list(arguments["hparams"].keys())

            # Now modify the inputed arguments accordingly
            for arg, value in input_args.items():
                if arg in hparams:
                    arguments["hparams"][arg] = value
                else:
                    arguments[arg] = value
        
        return arguments

    def load_model(self, model_path, dataset="train"):
        if not os.path.isfile(model_path):
            raise FileNotFoundError("The model_path specified does not exist")
        else:
            state_dict = torch.load(model_path)
            self.arguments = state_dict[1]
            # Correct for arguments added in the future
            if "sweep" not in self.arguments.keys():
                self.arguments["sweep"] = False
                
            self.model, self.dataset = prepare(self.arguments, state_dict[0], dataset)
    
    def train_fcr(self, state_dict=None):
        """
        Train an FCR model
        """
        args = self.arguments
        train(args, state_dict=state_dict)
            
        
        