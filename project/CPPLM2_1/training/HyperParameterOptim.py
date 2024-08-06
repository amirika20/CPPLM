# Directory configuration
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.dataloader import CPPDataloader
from training.trainer import CPPLMTrainer
from model.model import CPPLM

import torch
import torch.nn as nn
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
from functools import partial
from pathlib import Path

def train_CPPLM(config):
    # loading dataset
    dataset = CPPDataloader()
    train_loader, val_loader, test_loader = dataset.prep_data()

    # Building Model
    model = CPPLM(config["d_model"], config["n_heads"], config["n_layers"])

    # Building Trainder
    trainer = CPPLMTrainer(model)
    trainer.configure_optimizer()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)


    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["net_state_dict"])
            trainer.optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    trainer.train(train_loader, val_loader, test_loader)

    print("Finished Training")

def main(num_samples=5, max_num_epochs=20, gpus_per_trial=1):


    config = {
        "d_model": tune.choice([2**i for i in range(7,11)]),
        "n_heads": tune.choice([2**i for i in range(4)]),
        "n_layers": tune.choice([2**i for i in range(6)]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "gamma": tune.uniform(0.4,1),
        "batch_size": tune.choice([2, 4, 8, 16]),
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=5,
        reduction_factor=2,
    )
    result = tune.run(
        train_CPPLM,
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
    )
    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    # print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")
    return

main()