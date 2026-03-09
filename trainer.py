import logging
import sys
import time
from dataclasses import dataclass
import datetime
from pathlib import Path
from typing import List, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import torch
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from config import (
    AdaptiveAvgPool2dConfig,
    BatchNorm1dConfig,
    BatchNorm2dConfig,
    Config,
    Conv2dConfig,
    ConvTranspose2dConfig,
    Dropout1dConfig,
    FlattenConfig,
    LinearConfig,
    LossConfig,
    ModelConfig,
    ProjectionLossConfig,
    TrainingConfig,
    UnflattenConfig,
)
from data_loader import create_loaders_for_dataset
from loss_function import LossFn, create_loss_function, create_projection_loss, create_reconstruction_loss
from metrics import trustworthiness_continuity_powers_of_two
from models import EncoderDecoder, create_model, save_model
from utils import set_seed
from visual_eval import render_all_images

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(ch)

# Determine CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss_log_string(loss_name: str, values) -> str:
    return f"{loss_name} loss = {values[0]:.2f} (recon: {values[1]:.2f}, proj: {values[2]:.2f}, reg: {values[3]:.2f})"


@dataclass
class TrainingTask:
    type: Literal["encode", "decode", "full"]
    optimizer: Optimizer
    loss_func: LossFn


def training_step(train_loader: DataLoader, model: EncoderDecoder, tasks: List[TrainingTask]) -> torch.Tensor:
    model.train()
    losses = torch.zeros(4, device=device)

    for original, proj_target, _ in train_loader:
        original = original.to(device)
        proj_target = proj_target.to(device)

        for task in tasks:
            task.optimizer.zero_grad()
            reconstruction, latent, latent_mean, latent_covar = model(original, proj_target, mode=task.type)
            loss, recon, proj, reg = task.loss_func(
                original, proj_target, reconstruction, latent, latent_mean, latent_covar
            )
            loss.backward()
            losses += torch.tensor([loss.item(), recon.item(), proj.item(), reg.item()], device=device)
            task.optimizer.step()

    avg_losses = losses / len(train_loader)
    return avg_losses


def validation_test_step(val_loader: DataLoader, model: EncoderDecoder, tasks: List[TrainingTask]):
    model.eval()
    losses = torch.zeros(4,  device=device)

    with torch.no_grad():
        for original, proj_target, _ in val_loader:
            original = original.to(device)
            proj_target = proj_target.to(device)

            for task in tasks:
                reconstruction, latent, latent_mean, latent_covar = model(original, proj_target, mode=task.type)
                loss, recon, proj, reg = task.loss_func(
                    original, proj_target, reconstruction, latent, latent_mean, latent_covar
                )
                losses += torch.tensor([loss.item(), recon.item(), proj.item(), reg.item()], device=device)

    avg_losses = losses / len(val_loader)
    return avg_losses


def store_losses(losses_list: List[torch.Tensor], filename: str, time=None, epochs=None) -> None:
    if time is None or epochs is None:
        data = torch.stack(losses_list)
        df = pd.DataFrame(data.detach().cpu().numpy(), columns=["Loss", "Recon", "Proj", "Reg"])
        df.index.name = "Epoch"
    else:
        data = losses_list[0]
        data = torch.cat((data.detach().cpu(), torch.tensor([time]), torch.tensor([epochs])))
        df = pd.DataFrame(data.numpy().reshape(1, -1), columns=["Loss", "Recon", "Proj", "Reg", "Train", "Epochs"])
        df.index.name = "Id"
    df.to_csv(filename, index=True)


def store_trust_cont(
    ks: np.ndarray,
    trust: np.ndarray,
    cont: np.ndarray,
    filename: str,
) -> None:
    df = pd.DataFrame({
        "k": ks,
        "Trust": trust,
        "Continuity": cont,
    })

    df.to_csv(filename, index=False)


class ValidationSaveStop:
    def __init__(self, patience: int = 5, save: bool = True):
        self.patience = patience
        self.save = save
        self.counter = 0
        self.best_val_loss = float("inf")

    def check(self, val_loss, model):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.counter = 0
            if self.save:
                save_model(model)
                return "SAVED"
            return "IMPROVED"
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return "STOPPED"
            return "NONE"


def create_training_tasks(config: Config, model: EncoderDecoder):
    tasks: List[TrainingTask]
    if config.model.type == "pr":
        encoder_params = list(model.encoder_base.parameters()) + list(model.encoder_head.parameters())
        tasks = [
            TrainingTask(
                "encode", Adam(encoder_params, lr=config.training.learning_rate), create_projection_loss(config)
            ),
            TrainingTask(
                "decode",
                Adam(model.decoder.parameters(), lr=config.training.learning_rate),
                create_reconstruction_loss(config),
            ),
        ]
    else:
        tasks = [
            TrainingTask(
                "full", Adam(model.parameters(), lr=config.training.learning_rate), create_loss_function(config)
            )
        ]
    return tasks


@torch.no_grad()
def collect_test_spaces(test_loader, model, device):
    model.eval()

    X_high = []
    X_low  = []

    for original, _, _ in test_loader:
        x = original.to(device)
        z, mu, _ = model.encode(x)
        z = mu if mu is not None else z

        X_high.append(x.cpu())
        X_low.append(z.cpu())

    X_high = torch.cat(X_high, dim=0).numpy()
    X_low  = torch.cat(X_low,  dim=0).numpy()

    return X_high, X_low


def train(config: Config) -> EncoderDecoder:
    set_seed(config.training.seed)

    # Load dataset
    train_loader, val_loader, test_loader = create_loaders_for_dataset(
        config.dataset, config.projection, batch_size=config.training.batch_size
    )

    # Initialize model
    model = create_model(config).to(device)

    tasks = create_training_tasks(config, model)

    # Log configuration
    logger.info(config)

    # Log state dict
    message = "state_dict:\n"
    for param_tensor, param_value in model.state_dict().items():
        message += f"{param_tensor} \t {param_value.size()}\n"
    logger.info(message)

    #
    # Training loop
    #

    save_stop = ValidationSaveStop(patience=config.training.patience)
    train_records = []
    val_records = []

    start = time.time()

    for epoch in range(config.training.max_epochs):
        avg_train_losses = training_step(train_loader, model, tasks)
        train_records.append(avg_train_losses)
        train_log = loss_log_string("Train", avg_train_losses)

        avg_val_losses = validation_test_step(val_loader, model, tasks)
        val_records.append(avg_val_losses)
        val_log = loss_log_string("Val", avg_val_losses)

        status = save_stop.check(avg_val_losses[0], model)
        logger.info(f"Epoch {epoch + 1}: Status = {status}; {train_log}; {val_log}")
        if status == "STOPPED":
            break

    filename = Config.create_filename(model.config)
    training_time = time.time() - start
    logger.info(f"{filename}: Training time = {training_time} seconds")
    store_losses(train_records, f"./records/{filename}.train.csv")
    store_losses(val_records, f"./records/{filename}.val.csv")

    # Evaluate on test set
    avg_test_losses = validation_test_step(test_loader, model, tasks)
    test_log = loss_log_string("Test", avg_test_losses)
    logger.info(f"{filename}: {test_log}")
    store_losses([avg_test_losses], f"./records/{filename}.test.csv", time=training_time, epochs=(epoch + 1))


    X_high, X_low = collect_test_spaces(test_loader, model, device)

    D_high = pairwise_distances(X_high, metric="euclidean")
    D_low  = pairwise_distances(X_low,  metric="euclidean")
    
    ks, trust, cont = trustworthiness_continuity_powers_of_two(D_high, D_low)
    
    store_trust_cont(ks, trust, cont, f"./records/{filename}.truts-cont.csv")

    return model


def old_main():
    config = Config(
        dataset="fmnist",
        projection="tsne",
        comment="test",
        model=ModelConfig(
            type="vae",
            io_dim=28 * 28,
            latent_dim=2,
            encoder_layers=[
                LinearConfig(out_features=1024, activation="sigmoid"),
                LinearConfig(out_features=512, activation="sigmoid"),
                LinearConfig(out_features=256, activation="sigmoid"),
                LinearConfig(out_features=128, activation="sigmoid"),
                LinearConfig(out_features=64, activation="sigmoid"),
            ],
            decoder_layers=[
                LinearConfig(out_features=64, activation="sigmoid"),
                LinearConfig(out_features=128, activation="sigmoid"),
                LinearConfig(out_features=256, activation="sigmoid"),
                LinearConfig(out_features=512, activation="sigmoid"),
                LinearConfig(out_features=1024, activation="sigmoid"),
                LinearConfig(out_features=28 * 28, activation="sigmoid"),
            ],
        ),
        loss_recon=LossConfig(loss_fn="bce", weight=100.0),
        loss_proj=ProjectionLossConfig(target="latent", loss_fn="mse", weight=10.0),
        loss_reg=LossConfig(loss_fn="kl_div", weight=0.1),
        training=TrainingConfig(max_epochs=1000, batch_size=128, learning_rate=1e-4, patience=20, seed=777),
    )

    model = train(config)
    render_all_images(model)



def main():
    yaml_dir = Path("./af_yaml")
    records_dir = Path("./records")
    cutoff_date = datetime.datetime(2026, 1, 12)

    yaml_files = sorted(
        list(yaml_dir.glob("*.yaml")) +
        list(yaml_dir.glob("*.yml"))
    )

    for yaml_path in yaml_files:
        config = Config.load_from_yaml(str(yaml_path))
        output_filename = config.create_filename()
        test_file = records_dir / f"{output_filename}.test.csv"

        if test_file.exists():
            mod_time = datetime.datetime.fromtimestamp(test_file.stat().st_mtime)
            if mod_time > cutoff_date:
                print(f"Skipping {yaml_path.name} (already trained)")
                continue

        print(f"Training: {yaml_path.name}")
        model = train(config)
        render_all_images(model)


if __name__ == "__main__":
    main()
