import os
import time
import logging
from datetime import datetime
from datetime import timedelta
from collections import defaultdict
import wandb
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from socket import gethostname
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from ..evaluate.evaluate import evaluate, evaluate_classic,evaluate_prediction

from ..model.model_parallel import load_FCR


from ..dataset.dataset_parallel import load_dataset_splits,load_dataset_train_test, get_dataset_features

from ..utils.general_utils import initialize_logger, ljson
from ..utils.data_utils import data_collate

from ..validation import plot_umaps
# from ..validation import plot_progression

torch.autograd.set_detect_anomaly(True)

## MODIFIED: add split to select desired datastet
## MODIFIED: we will repeteadly load shards or partitions of an original dataset
def prepare(args, features, shard_path):
    """
    Instantiates dataset (partition).
    """
    
    # dataset
    if args['covariate_keys']!= None:
        covariate_keys = args['covariate_keys']
    else:
        covariate_keys = 'covariates'

    if args['perturbation_key']!=None:
        perturbation_key = args["perturbation_key"]
    else:
        perturbation_key = "Agg_Treatment"
        
    if args['split']!= None:
        split_key = args["split"]
    
    # Modified: compatibility with Tahoe100M plates
    control_name = args.get("control_name", None)
    embedded_dose = args.get("embedded_dose", None)

    datasets = load_dataset_train_test(
    shard_path,
    args,
    features,
    perturbation_input = args.get("perturbation_input", "ohe"),
    covariate_keys = covariate_keys,
    perturbation_key = perturbation_key,
    split_key = args["split_key"],
    sample_cf=(True if args["dist_mode"] == "match" else False),
    control_name = control_name,
    embedded_dose = embedded_dose,
    )

    return datasets

# FUNCTION TO SET UP PROCESS
def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=60))


def train(args, prepare=prepare, state_dict=None):
    """
    Trains a FCR model
    """
    if args["seed"] is not None:
        np.random.seed(args["seed"])
        torch.manual_seed(args["seed"])

    # SETUP FOR PARALLEL COMPUTING
    use_cuda = args["gpu"]

    world_size = int(os.environ["WORLD_SIZE"])
    rank  = int(os.environ["SLURM_PROCID"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])

    assert gpus_per_node == torch.cuda.device_count()
    print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
          f" {gpus_per_node} allocated GPUs per node.", flush=True)
    
    setup(rank, world_size)
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

    # Only the Rank 0 process will log
    log_proc = (rank == 0)

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")

    # Get global dataste features
    if log_proc:
        print("Obtaining global dataset features...")

    features = {}
    control_names, var_names, pert_unique, covars_dict, cf_genemap, num_treatments, num_outcomes, num_covariates = \
    get_dataset_features(
        args["data_path"],
        covariate_keys = args["covariate_keys"],
        perturbation_key = args["perturbation_key"],
        perturbation_input = args["perturbation_input"],
        control_key= args["control_key"]
    )
    features["control_names"] = control_names
    features["var_names"] = var_names
    features["pert_unique"] = pert_unique
    features["covars_dict"] = covars_dict
    features["cf_genemap"] = cf_genemap

    # Load Model
    args["num_outcomes"] = num_outcomes
    # print(f"Num_treatments: {num_treatments}")
    args["num_treatments"] = num_treatments
    args["num_covariates"] = num_covariates

    model = load_FCR(args, state_dict)

    args["hparams"] = model.hparams

    # Setup DDP model
    model.to(local_rank)
    model.device = torch.device(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer_autoencoder = optim.Adam(ddp_model.module.params_autoencoder,
                                       lr=args["hparams"]["autoencoder_lr"],
                                       weight_decay=args["hparams"]["autoencoder_wd"]
                                       )
    scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(
        optimizer_autoencoder, 
        step_size=args["hparams"]["step_size_lr"],
        gamma=0.1
    )

    optimizer_discriminator = optim.Adam(ddp_model.module.params_discriminator,
                                       lr=args["hparams"]["discriminator_lr"],
                                       weight_decay=args["hparams"]["discriminator_wd"]
                                       )
    scheduler_discriminator = torch.optim.lr_scheduler.StepLR(
        optimizer_discriminator,
        step_size=args["hparams"]["step_size_lr"],
        gamma=0.1
    )

    """
    Train the FCR model with DDP
    """
    model = ddp_model

    # Setup logging (only rank 0 process log)
    if log_proc:
        wandb.init(config=args, project=args["name"], name=args["experiment"])
        # wandb.watch(model, log_freq=10) # Disabled for the moment for DDP setup

        dt = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
        writer = SummaryWriter(log_dir=os.path.join(args["artifact_path"], "runs/" + args["name"] + "_" + dt))
        save_dir = os.path.join(args["artifact_path"], "saves/" + args["name"] + "_" + dt)
        os.makedirs(save_dir, exist_ok=True)

        initialize_logger(save_dir)
        ljson({"training_args": args})
        ljson({"model_params": model.module.hparams})
        logging.info("")

    """
    TRAINING LOOP:
    - For each epoch:
        - For each shard/partition of the dataset:
            - Load shard
            - Set Distributed DataLoader
            - For each minibatch:
                - Forward pass
                - Backward pass
                - Optimization step
            - Log shard_training_stats each {checkpoint_frequency} shards

        - Save model checkpoint each epoch
        - Plot UMAPs for each model checkpoint
    """

    start_time = time.time()

    # Determine number of shards
    shard_path = Path(args["data_path"]).parent
    shard_count = len(list(shard_path.glob("adata_part*")))

    start_epoch = args.get("epoch", 0)  # in case the training is resumed from a checkpoint
    print(f"Starting training from epoch {start_epoch}")

    stop = False
    for epoch in range(start_epoch, args["max_epochs"]):

        # Randomly shuffle shard order
        rng = np.random.default_rng(args.get("seed", 0) + epoch)
        rand_shard_ids = rng.permutation(shard_count)
    
        # Determine epoch time
        epoch_start_time = time.time()

        # Check stopping conditions
        if stop:
            break

        # Go over gene expression shards
        for shard_id in range(shard_count):
            # Choose random shard index
            rand_shard_id = rand_shard_ids[shard_id]

            # All ranks start assuming no error
            error_flag = torch.zeros(1, device=local_rank)

            # SHARD PREPARATION WITH ERROR SYNCRONIZATION ACROSS RANKS
            try: 
                shard_training_stats = defaultdict(float)
                shard_start_time = time.time()

                if log_proc:
                    print(f"Loading shard {shard_id} (id {rand_shard_id}) for epoch {epoch}...")

                # Update gene expression shard for this epoch
                shard_file = shard_path / f"adata_part{rand_shard_id}.h5ad"
                datasets = prepare(args, features, shard_file)

                # Set Distributed DataLoader
                # Train Sampler ensures no overlapping samples between processes
                train_sampler = torch.utils.data.distributed.DistributedSampler(datasets["train"],
                                                                                num_replicas=world_size,
                                                                                rank=rank,
                                                                                shuffle=True
                                                                                )
                
                loader =  torch.utils.data.DataLoader(
                            datasets["train"],
                            batch_size=args["batch_size"],
                            sampler=train_sampler,
                            num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                            prefetch_factor=4,
                            pin_memory = True,
                            persistent_workers=False,
                            drop_last=True,
                            collate_fn=(lambda batch: data_collate(batch, nb_dims=1))
                        )

                # Set epoch for DistributedSampler to reshuffle differently each epoch
                loader.sampler.set_epoch(epoch)
                
                # Set adversarial training flag
                # if (epoch % args["adv_epoch"]) == 0 and epoch > 0:
                if (shard_id % args["adv_epoch"] == 0) and shard_id > 0:
                    adv_training=True
                else:
                    adv_training=False
                # print("Adversarial Training {}".format(adv_training))

            except Exception as e:
                print(f"[Prep error] Rank {rank} failed shard {rand_shard_id}: {e}")
                error_flag[0] = 1

            # Sync shard prep status
            dist.all_reduce(error_flag, op=dist.ReduceOp.SUM)
            if error_flag.item() > 0:
                if log_proc:
                    print(f"Skipping shard {rand_shard_id} on all ranks due to prep error.")
                dist.barrier()
                continue

            # MODEL TRAINING FOR CURRENT SHARD
            model.train()
            minibatch_counter = 0
            for data in loader:

                # print("Training with minibatch ", minibatch_counter)
                (experiment, treatment, control, _, covariates)= \
                (data[0], data[1], data[2], data[3], data[4:])

                # Freeze the discriminator if not adv_training
                if not adv_training:
                    model.module.freeze_discriminator(True)

                # Forward pass through DDP wrapper
                loss, minibatch_training_stats = \
                model(
                    experiment, 
                    treatment, 
                    control, 
                    covariates, 
                    adv_training=adv_training, 
                    sample_latent=args["hparams"]["sample_latent"],
                    single_treatment=args.get("single_treatment", False)
                )
                
                # Backward pass and optimization step
                optimizer_autoencoder.zero_grad()
                optimizer_discriminator.zero_grad()
                
                # Single backward pass
                loss.backward()
                
                # Step only the relevant optimizer based on training phase
                if not adv_training:
                    optimizer_autoencoder.step()
                else:
                    optimizer_discriminator.step()

                # Unfreeze discriminator after iteration
                if not adv_training:
                    model.module.freeze_discriminator(False)

                minibatch_counter += 1

                # Logging minibatches
                if minibatch_counter == 3 and log_proc:
                    print(f"Epoch {epoch} - Shard {rand_shard_id} - Minibatch {minibatch_counter}")
                    sec_per_mill_gpu = 1e6/((minibatch_counter * args['batch_size'])/(time.time() - shard_start_time))
                    hour_per_mill_gpu = sec_per_mill_gpu / 3600
                    print(f"1M samples rate (1 GPU): {hour_per_mill_gpu} hours")

                for key, val in minibatch_training_stats.items():
                    shard_training_stats[key] += val

            # Average shard stats over number of minibatches
            for key, val in shard_training_stats.items():
                shard_training_stats[key] = val / len(loader)
                if not (key in model.module.history.keys()):
                    model.module.history[key] = []
                model.module.history[key].append(shard_training_stats[key])

            if not "epoch_shard" in model.module.history.keys():
                model.module.history["epoch_shard"] = []
            model.module.history["epoch_shard"].append(f"{epoch}_{shard_id}")

            ellapsed_minutes = (time.time() - start_time) / 60
            model.module.history["elapsed_time_min"] = ellapsed_minutes

            # Only rank 0 logs
            if ((shard_id % args["checkpoint_freq"]) == 0):
                # print("Performing evaluation...")
                # Activate evaluation mode
                model.eval()

                evaluation_stats = evaluate_prediction(model.module, datasets, args)
                for key, val in evaluation_stats.items():
                    if not (key in model.module.history.keys()):
                        model.module.history[key] = []
                    model.module.history[key].append(val)

                if log_proc:
                    # Log stats to console and file
                    ljson(
                        {
                            "epoch": epoch,
                            "training_stats": shard_training_stats,
                            "evaluation_stats": evaluation_stats,
                            "ellapsed_minutes": ellapsed_minutes,
                            "Discriminator Training": adv_training
                        }
                    )

                    # Log stats to WandB
                    all_stats = {}
                    for stat, value in shard_training_stats.items():
                        all_stats[stat] = value
                    
                    for stat, value in evaluation_stats.items():
                        all_stats[stat] = value

                    all_stats["ellapsed_minutes"] = ellapsed_minutes
                    all_stats["R2 Score Train (Mean)"] = evaluation_stats["train"][0]
                    all_stats["R2 Score Train (Stddev)"] = evaluation_stats["train"][1]
                    all_stats["R2 Score Test (Mean)"] = evaluation_stats["test"][0]
                    all_stats["R2 Score Test (Stddev)"] = evaluation_stats["test"][1]

                    wandb.log(all_stats)

                    for key, val in shard_training_stats.items():
                        writer.add_scalar(key, val, epoch)

                # Step schedulers once per (checkpoint_freq * step_size_lr) shards and check early stopping
                model.module.early_stopping(shard_training_stats["KL Divergence"], scheduler_autoencoder, scheduler_discriminator)
                # TEMPORARILY DISABLED EARLY STOP
                # stop = stop or model.module.early_stopping(shard_training_stats["KL Divergence"], scheduler_autoencoder, scheduler_discriminator)
                # if stop:
                #     ljson({"early_stop": epoch})
                #     break

            # Syncronize ranks before next shard
            dist.barrier()

            # Clean cache for next shard
            del loader, train_sampler
            # torch.cuda.empty_cache()


        # Fence rank 0 operations
        dist.barrier()
        # LOAD MODEL CHECKPOINT EACH EPOCH (Only rank 0 does)
        if log_proc:
            torch.save(
                (model.module.state_dict(), args, model.module.history),
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

            # UMAP PLOTTING EACH EPOCH
            print(f"Plotting UMAPs for epoch {epoch}!")
            plot_raw = True if epoch == 0 else False
            # plot_umaps(model_dir=args["artifact_path"], n_checkpoint=epoch, plot_raw=plot_raw, all_drugs=False, sample=False)
            # plot_progression(model_dir=args["artifact_path"], rep="ZXs", feature="cell_name", freq=100)
            # plot_progression(model_dir=args["artifact_path"], rep="ZTs", feature="dose", freq=100)

        dist.barrier()

        # Determine if last epoch
        stop = stop or (epoch == args["max_epochs"] - 1)

    if log_proc:
        writer.close()

    # Cleanup the distributed backend
    dist.destroy_process_group()
 

