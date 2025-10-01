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


class FCR_sim:
    def __init__(self, config_path=None, model_path=None, dataset_mode="train"):
        # Importing mode for dataset
        self.dataset_mode = dataset_mode
        if model_path != None:
            self.load_model(model_path, self.dataset_mode)
        elif config_path != None:
            self.arguments = self.parse_arguments(config_path)
        else:
            raise ValueError("Valid config file or model path required")

    def parse_arguments(self, path: str):
        """
        Read arguments from config file
        """
        with open(path, "r") as f:
            return json.load(f)

    def load_model(self, model_path, dataset="train"):
        if not os.path.isfile(model_path):
            raise FileNotFoundError("The model_path specified does not exist")
        else:
            state_dict = torch.load(model_path)
            self.arguments = state_dict[1]
            self.model, self.dataset = prepare(self.arguments, state_dict[0], dataset)

    def train_fcr(self, state_dict=None):
        """
        Train an FCR model
        """
        args = self.arguments
        train(args, state_dict=state_dict)

    def fetch_latest(self, suffix=".pt"):
        saves_dir = os.path(self.arguments["artifact_path"],"saves")
        if not os.path.isdir(saves_dir):
            raise FileNotFoundError(f"Directory not found: {saves_dir}")
            
        ckpts = [os.path.join(saves_dir, f) for f in os.listdir(saves_dir) if f.endswith(suffix)]
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint files with suffix '{suffix}' found in {saves_dir}")
        
        latest_ckpt = max(ckpts, key=os.path.getmtime)
        return latest_ckpt
            
        
        