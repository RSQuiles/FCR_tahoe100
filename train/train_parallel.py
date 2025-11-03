import os
import time
import logging
from datetime import datetime
from collections import defaultdict
import wandb
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from socket import gethostname

from torch.utils.tensorboard import SummaryWriter

from ..evaluate.evaluate import evaluate, evaluate_classic,evaluate_prediction

from ..model.model_parallel import load_FCR


from ..dataset.dataset import load_dataset_splits,load_dataset_train_test

from ..utils.general_utils import initialize_logger, ljson
from ..utils.data_utils import data_collate

from ..validation import plot_umaps
from ..validation import plot_progression

## modified: add split to select desired datastet
def prepare(args, world_size, rank, local_rank, state_dict=None, split_name="train"):
    """
    Instantiates model and dataset to run an experiment.
    """

#     perturbation_key = "perturbation",
#     control_key = "control",
#         dose_key = "dose",
#         covariate_keys = "cell_type",
#         split_key = "split"
    
    
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

    # if args['split']=="split":
    #     datasets = load_dataset_splits(
    #         args["data_path"],
    #         sample_cf=(True if args["dist_mode"] == "match" else False),
    #     )
    # elif args['split']=="new_split":
    datasets = load_dataset_train_test(
    args["data_path"],
    perturbation_input = args.get("perturbation_input", "ohe"),
    covariate_keys = covariate_keys,
    perturbation_key = perturbation_key,
    split_key = args["split_key"],
    sample_cf=(True if args["dist_mode"] == "match" else False),
    control_name = control_name,
    embedded_dose = embedded_dose,
    args = args,
    )

    # Train Sampler ensures no overlapping samples between processes
    train_sampler = torch.utils.data.distributed.DistributedSampler(datasets[split_name],
                                                                    num_replicas=world_size,
                                                                    rank=rank,
                                                                    shuffle=False
                                                                    )
    
    datasets.update(
        {
            "loader_tr": torch.utils.data.DataLoader(
                datasets[split_name],
                batch_size=args["batch_size"],
                sampler=train_sampler,
                num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                collate_fn=(lambda batch: data_collate(batch, nb_dims=1))
            )
        }
    )

    args["num_outcomes"] = datasets["train"].num_outcomes
    args["num_treatments"] = datasets["train"].num_treatments
    # print(f"num_treatments: {args['num_treatments']}")
    args["num_covariates"] = datasets["train"].num_covariates

    # model
    model = load_FCR(args, state_dict)
    model.to(local_rank)
    # print("load FCR model")

    args["hparams"] = model.hparams

    return model, datasets

# FUNCTION TO SET UP PROCESS
def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train_ddp(model, args, datasets, log_proc, 
              optimizer_autoencoder, optimizer_discriminator,
              scheduler_autoencoder, scheduler_discriminator):
    """
    Trains a FCR model with DDP
    """
    # Setup logging (only rank 0 process log)
    if log_proc:
        wandb.init(config=args, project=args["name"], name=args["experiment"])
        wandb.watch(model, log_freq=10)

        dt = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
        writer = SummaryWriter(log_dir=os.path.join(args["artifact_path"], "runs/" + args["name"] + "_" + dt))
        save_dir = os.path.join(args["artifact_path"], "saves/" + args["name"] + "_" + dt)
        os.makedirs(save_dir, exist_ok=True)

        initialize_logger(save_dir)
        ljson({"training_args": args})
        ljson({"model_params": model.hparams})
        logging.info("")

    start_time = time.time()

    for epoch in range(args["max_epochs"]):
        
        epoch_training_stats = defaultdict(float)
        if epoch % args["adv_epoch"]==0:
            adv_training=True
        else:
            adv_training=False
        # print("Adversarial Training {}".format(adv_training))

        minibatch_counter = 0
        for data in datasets["loader_tr"]:

            # print("Training with minibatch ", minibatch_counter)
            (experiment, treatment, control, _, covariates)= \
            (data[0], data[1], data[2], data[3], data[4:])

            # Check dimensions of inputs
            # print("Experiment dimensions: ", experiment.shape)
            # print("Treatment dimensions: ", treatment.shape)
            # print("Control dimensions: ", control.shape)
            # print("Covariates dimensions: ", [cov.shape for cov in covariates])

            minibatch_training_stats = model.update(
                experiment, treatment, control, covariates, optimizer_autoencoder, optimizer_discriminator, adv_training
            )
            
            minibatch_counter += 1
            
            ## Legacy code for training with divergence 
            # minibatch_training_stats = model.update_divergence(
            #     experiment, treatment, control, covariates, adv_training)

            for key, val in minibatch_training_stats.items():
                epoch_training_stats[key] += val
        model.update_eval_encoder()

        for key, val in epoch_training_stats.items():
            epoch_training_stats[key] = val / len(datasets["loader_tr"])
            if not (key in model.history.keys()):
                model.history[key] = []
            model.history[key].append(epoch_training_stats[key])
        model.history["epoch"].append(epoch)

        ellapsed_minutes = (time.time() - start_time) / 60
        model.history["elapsed_time_min"] = ellapsed_minutes

        # decay learning rate if necessary
        # also check stopping condition: 
        # patience ran out OR max epochs reached
        stop = (epoch == args["max_epochs"] - 1)

        if (epoch % args["checkpoint_freq"]) == 0 or stop:
            evaluation_stats = evaluate_prediction(model, datasets)
            for key, val in evaluation_stats.items():
                if not (key in model.history.keys()):
                    model.history[key] = []
                model.history[key].append(val)
            model.history["stats_epoch"].append(epoch)

            # Only rank 0 process logs and saves model
            if log_proc:
                ljson(
                    {
                        "epoch": epoch,
                        "training_stats": epoch_training_stats,
                        "evaluation_stats": evaluation_stats,
                        "ellapsed_minutes": ellapsed_minutes,
                        "Discriminator Training": adv_training
                    }
                )

                # Log stats to WandB
                all_stats = {}
                for stat, value in epoch_training_stats.items():
                    all_stats[stat] = value
                
                for stat, value in evaluation_stats.items():
                    all_stats[stat] = value

                all_stats["ellapsed_minutes"] = ellapsed_minutes
                all_stats["R2 Score Train (Mean)"] = evaluation_stats["train"][0]
                all_stats["R2 Score Train (Stddev)"] = evaluation_stats["train"][1]
                all_stats["R2 Score Test (Mean)"] = evaluation_stats["test"][0]
                all_stats["R2 Score Test (Stddev)"] = evaluation_stats["test"][1]

                wandb.log(all_stats)

                for key, val in epoch_training_stats.items():
                    writer.add_scalar(key, val, epoch)

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

            # Step schedulers and check early stopping
            stop = stop or model.early_stopping(evaluation_stats["test"][0], scheduler_autoencoder, scheduler_discriminator)
            if stop:
                ljson({"early_stop": epoch})
                break

     # Rank 0 plots UMAPS
    if log_proc:
        plot_umaps(model_dir=args["artifact_path"], all_drugs=False)
        # plot_progression(model_dir=args["artifact_path"], rep="ZXs", feature="cell_name", freq=100)
        # plot_progression(model_dir=args["artifact_path"], rep="ZTs", feature="dose", freq=100)

        writer.close()

    return 


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

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")

    # Load Model and Datasets
    if state_dict!=None:
        model, datasets = prepare(args, world_size, rank, local_rank, state_dict)  
    else:
        model, datasets = prepare(args, world_size, rank, local_rank)

    # Setup DDP model
    ddp_model = DDP(model, device_ids=[local_rank])

    optimizer_autoencoder = optim.Adam(ddp_model.module.params_autoencoder,
                                       lr=args["learning_rate"],
                                       weight_decay=args["hparams"]["autoencoder_wd"]
                                       )
    scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(
        optimizer_autoencoder, step_size=args["hparams"]["step_size_lr"]
    )

    optimizer_discriminator = optim.Adam(ddp_model.module.params_discriminator,
                                       lr=args["learning_rate"],
                                       weight_decay=args["hparams"]["discriminator_wd"]
                                       )
    scheduler_discriminator = torch.optim.lr_scheduler.StepLR(
        optimizer_discriminator, step_size=args["hparams"]["step_size_lr"]
    )

    # Only the Rank 0 process will log
    log_proc = (rank == 0)

    train_ddp(ddp_model, args, datasets, log_proc, 
              optimizer_autoencoder, optimizer_discriminator,
              scheduler_autoencoder, scheduler_discriminator)

    return

