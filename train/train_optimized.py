"""
Optimized training loop for large-scale FCR training

Key optimizations:
1. Mixed precision training (AMP)
2. Gradient accumulation
3. Optimized data loading
4. Memory-efficient operations
5. Better logging and profiling
"""

import os
import time
import logging
from datetime import datetime
from collections import defaultdict
import wandb

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
# from torch.nn.parallel import DistributedDataParallel as DDP

from ..evaluate.evaluate import evaluate, evaluate_classic, evaluate_prediction
from ..model import load_FCR
from ..dataset.dataset import load_dataset_splits, load_dataset_train_test
from ..utils.general_utils import initialize_logger, ljson
from ..utils.data_utils import data_collate
from ..validation import plot_umaps
from ..validation import plot_progression


def prepare_optimized(args, state_dict=None, split_name="train"):
    """
    Instantiates model and dataset with optimized data loading.
    """
    # Dataset configuration
    if args['covariate_keys'] != None:
        covariate_keys = args['covariate_keys']
    else:
        covariate_keys = 'covariates'

    if args['perturbation_key'] != None:
        perturbation_key = args["perturbation_key"]
    else:
        perturbation_key = "Agg_Treatment"

    control_name = args.get("control_name", None)
    embedded_dose = args.get("embedded_dose", None)

    # Load datasets
    datasets = load_dataset_train_test(
        args["data_path"],
        perturbation_input=args.get("perturbation_input", "ohe"),
        covariate_keys=covariate_keys,
        perturbation_key=perturbation_key,
        split_key=None,
        sample_cf=(True if args["dist_mode"] == "match" else False),
        control_name=control_name,
        embedded_dose=embedded_dose,
        args=args,
    )

    # Optimized DataLoader settings
    num_workers = args.get("num_workers", 4)
    pin_memory = args.get("pin_memory", True)
    prefetch_factor = args.get("prefetch_factor", 2)
    
    datasets.update(
        {
            "loader_tr": torch.utils.data.DataLoader(
                datasets[split_name],
                batch_size=args["batch_size"],
                shuffle=False,  # Important for large datasets
                collate_fn=(lambda batch: data_collate(batch, nb_dims=1)),
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                persistent_workers=True if num_workers > 0 else False,
            )
        }
    )

    args["num_outcomes"] = datasets["train"].num_outcomes
    args["num_treatments"] = datasets["train"].num_treatments
    args["num_covariates"] = datasets["train"].num_covariates

    # Load model
    model = load_FCR(args, state_dict)
    args["hparams"] = model.hparams

    return model, datasets


def train_optimized(args, prepare=prepare_optimized, state_dict=None):
    """
    Optimized training loop for large-scale FCR training.
    """
    if args["seed"] is not None:
        np.random.seed(args["seed"])
        torch.manual_seed(args["seed"])

    # Gradient accumulation settings
    accumulation_steps = args.get("gradient_accumulation_steps", 1)
    use_amp = args.get("use_mixed_precision", True)
    
    # Load Model and Datasets
    if state_dict != None:
        model, datasets = prepare(args, state_dict)
    else:
        model, datasets = prepare(args)

    # Initialize mixed precision scaler
    scaler = GradScaler() if use_amp else None

    # Setup WandB logging
    with wandb.init(config=args, project=args["name"], name=args["experiment"]) as run:
        
        # WandB tracking
        run.watch(model, log_freq=10)

        dt = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
        writer = SummaryWriter(log_dir=os.path.join(args["artifact_path"], "runs/" + args["name"] + "_" + dt))
        save_dir = os.path.join(args["artifact_path"], "saves/" + args["name"] + "_" + dt)
        os.makedirs(save_dir, exist_ok=True)

        initialize_logger(save_dir)
        ljson({"training_args": args})
        ljson({"model_params": model.hparams})
        ljson({"optimization": {
            "mixed_precision": use_amp,
            "gradient_accumulation_steps": accumulation_steps,
            "effective_batch_size": args["batch_size"] * accumulation_steps
        }})
        logging.info("")

        start_time = time.time()
        
        # Initialize gradient accumulation
        model.optimizer_autoencoder.zero_grad()
        model.optimizer_discriminator.zero_grad()
        
        for epoch in range(args["max_epochs"]):
            
            epoch_training_stats = defaultdict(float)
            epoch_start_time = time.time()
            
            # Adversarial training schedule
            if epoch % args["adv_epoch"] == 0:
                adv_training = True
            else:
                adv_training = False

            minibatch_counter = 0
            samples_processed = 0
            
            for batch_idx, data in enumerate(datasets["loader_tr"]):

                (experiment, treatment, control, _, covariates) = \
                    (data[0], data[1], data[2], data[3], data[4:])

                # Mixed precision training
                if use_amp:
                    with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                        # Model update returns loss stats
                        minibatch_training_stats = model.update(
                            experiment, treatment, control, covariates, adv_training
                        )
                else:
                    minibatch_training_stats = model.update(
                        experiment, treatment, control, covariates, adv_training
                    )

                """
                # Gradient accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    # Step optimizers
                    if use_amp:
                        scaler.step(model.optimizer_autoencoder)
                        if adv_training:
                            scaler.step(model.optimizer_discriminator)
                        scaler.update()
                    else:
                        model.optimizer_autoencoder.step()
                        if adv_training:
                            model.optimizer_discriminator.step()
                    
                    # Zero gradients
                    model.optimizer_autoencoder.zero_grad()
                    model.optimizer_discriminator.zero_grad()
                """

                minibatch_counter += 1
                samples_processed += experiment.shape[0]

                # Accumulate stats
                for key, val in minibatch_training_stats.items():
                    epoch_training_stats[key] += val

                # Log throughput periodically
                if minibatch_counter % 100 == 0:
                    epoch_time = time.time() - epoch_start_time
                    samples_per_sec = samples_processed / epoch_time
                    logging.info(f"Epoch {epoch} - Batch {minibatch_counter}: {samples_per_sec:.1f} samples/sec")

                    # Log GPU memory usage periodically
                    if torch.cuda.is_available():
                        mem_allocated = torch.cuda.memory_allocated() / 1e9
                        mem_cached = torch.cuda.memory_reserved() / 1e9
                        logging.info(f"Batch {minibatch_counter}: GPU Memory: {mem_allocated:.2f}GB allocated, {mem_cached:.2f}GB cached")

            # Update eval encoder
            model.update_eval_encoder()

            # Average epoch stats
            for key, val in epoch_training_stats.items():
                epoch_training_stats[key] = val / len(datasets["loader_tr"])
                if not (key in model.history.keys()):
                    model.history[key] = []
                model.history[key].append(epoch_training_stats[key])
            
            model.history["epoch"].append(epoch)

            ellapsed_minutes = (time.time() - start_time) / 60
            epoch_time = time.time() - epoch_start_time
            model.history["elapsed_time_min"] = ellapsed_minutes
            
            # Calculate throughput
            samples_per_sec = samples_processed / epoch_time
            
            # Logging
            logging.info(f"Epoch {epoch}: {samples_per_sec:.1f} samples/sec, {epoch_time:.1f}s total")

            # Stop condition
            stop = (epoch == args["max_epochs"] - 1)

            # Evaluation and checkpointing
            if (epoch % args["checkpoint_freq"]) == 0 or stop:
                evaluation_stats = evaluate_prediction(model, datasets)
                
                for key, val in evaluation_stats.items():
                    if not (key in model.history.keys()):
                        model.history[key] = []
                    model.history[key].append(val)
                model.history["stats_epoch"].append(epoch)

                ljson(
                    {
                        "epoch": epoch,
                        "training_stats": epoch_training_stats,
                        "evaluation_stats": evaluation_stats,
                        "ellapsed_minutes": ellapsed_minutes,
                        "samples_per_sec": samples_per_sec,
                        "discriminator_training": adv_training
                    }
                )

                # Log stats to WandB
                all_stats = {}
                for stat, value in epoch_training_stats.items():
                    all_stats[stat] = value
                
                for stat, value in evaluation_stats.items():
                    all_stats[stat] = value

                all_stats["ellapsed_minutes"] = ellapsed_minutes
                all_stats["samples_per_sec"] = samples_per_sec
                all_stats["epoch_time_sec"] = epoch_time
                all_stats["R2 Score Train (Mean)"] = evaluation_stats["train"][0]
                all_stats["R2 Score Train (Stddev)"] = evaluation_stats["train"][1]
                all_stats["R2 Score Test (Mean)"] = evaluation_stats["test"][0]
                all_stats["R2 Score Test (Stddev)"] = evaluation_stats["test"][1]

                run.log(all_stats)

                for key, val in epoch_training_stats.items():
                    writer.add_scalar(key, val, epoch)

                # Save checkpoint
                torch.save(
                    (model.state_dict(), args, model.history),
                    os.path.join(
                        save_dir,
                        "model_seed={}_epoch={}.pt".format(args["seed"], epoch),
                    ),
                )

                ljson(
                    {
                        "model_saved": "model_seed={}_epoch={}.pt\n".format(
                            args["seed"], epoch
                        )
                    }
                )
                
                stop = stop or model.early_stopping(evaluation_stats["test"][0])
                if stop:
                    ljson({"early_stop": epoch})
                    break

        # Final visualizations
        plot_umaps(model_dir=args["artifact_path"], all_drugs=False)

        writer.close()
        return model


if __name__ == "__main__":
    import json
    import sys
    
    # Load config from command line
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            args = json.load(f)
        
        # Set optimization defaults
        args.setdefault("use_mixed_precision", True)
        args.setdefault("gradient_accumulation_steps", 4)
        args.setdefault("num_workers", 4)
        args.setdefault("pin_memory", True)
        args.setdefault("prefetch_factor", 2)
        
        # Train
        model = train_optimized(args)
    else:
        print("Usage: python train_optimized.py config.json")
