import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import pairwise_distances
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from config import Config
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
    losses = torch.zeros(4, device=device)

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
    df = pd.DataFrame(
        {
            "k": ks,
            "Trust": trust,
            "Continuity": cont,
        }
    )

    df.to_csv(filename, index=False)


class ValidationSaveStop:
    def __init__(self, patience: int = 5, save: bool = True, models_dir: str = "models"):
        self.patience = patience
        self.save = save
        self.models_dir = models_dir
        self.counter = 0
        self.best_val_loss = float("inf")

    def check(self, val_loss, model):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.counter = 0
            if self.save:
                save_model(model, self.models_dir)
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
    X_low = []

    for original, _, _ in test_loader:
        x = original.to(device)
        z, mu, _ = model.encode(x)
        z = mu if mu is not None else z

        X_high.append(x.cpu())
        X_low.append(z.cpu())

    X_high = torch.cat(X_high, dim=0).numpy()
    X_low = torch.cat(X_low, dim=0).numpy()

    return X_high, X_low


def train(
    config: Config,
    records_dir: str = "./records",
    models_dir: str = "models",
    preprocessed_dir: str = "./preprocessed",
) -> EncoderDecoder:
    set_seed(config.training.seed)

    # Load dataset
    train_loader, val_loader, test_loader = create_loaders_for_dataset(
        config.dataset, config.projection, batch_size=config.training.batch_size, preprocessed_dir=preprocessed_dir
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

    save_stop = ValidationSaveStop(patience=config.training.patience, models_dir=models_dir)
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
    records = Path(records_dir)
    records.mkdir(parents=True, exist_ok=True)
    logger.info(f"{filename}: Training time = {training_time} seconds")
    store_losses(train_records, str(records / f"{filename}.train.csv"))
    store_losses(val_records, str(records / f"{filename}.val.csv"))

    # Evaluate on test set
    avg_test_losses = validation_test_step(test_loader, model, tasks)
    test_log = loss_log_string("Test", avg_test_losses)
    logger.info(f"{filename}: {test_log}")
    store_losses([avg_test_losses], str(records / f"{filename}.test.csv"), time=training_time, epochs=(epoch + 1))

    X_high, X_low = collect_test_spaces(test_loader, model, device)

    D_high = pairwise_distances(X_high, metric="euclidean")
    D_low = pairwise_distances(X_low, metric="euclidean")

    ks, trust, cont = trustworthiness_continuity_powers_of_two(D_high, D_low)

    store_trust_cont(ks, trust, cont, str(records / f"{filename}.truts-cont.csv"))

    return model


def main():
    parser = argparse.ArgumentParser(description="Train models from YAML configs.")
    parser.add_argument("--config-dir", type=Path, default=Path("./configs"), help="Directory of experiment YAMLs.")
    parser.add_argument("--records-dir", type=Path, default=Path("./records"), help="Output dir for training records.")
    parser.add_argument("--models-dir", type=Path, default=Path("models"), help="Output dir for checkpoints + configs.")
    parser.add_argument(
        "--preprocessed-dir", type=Path, default=Path("./preprocessed"), help="Dir of 2D projection targets."
    )
    parser.add_argument("--force", action="store_true", help="Retrain even if a test record already exists.")
    parser.add_argument("--no-render", action="store_true", help="Skip rendering evaluation images after training.")
    args = parser.parse_args()

    yaml_files = sorted(list(args.config_dir.glob("*.yaml")) + list(args.config_dir.glob("*.yml")))

    for yaml_path in yaml_files:
        config = Config.load_from_yaml(str(yaml_path))
        test_file = args.records_dir / f"{config.create_filename()}.test.csv"

        if test_file.exists() and not args.force:
            print(f"Skipping {yaml_path.name} (already trained; use --force to retrain)")
            continue

        print(f"Training: {yaml_path.name}")
        model = train(
            config,
            records_dir=str(args.records_dir),
            models_dir=str(args.models_dir),
            preprocessed_dir=str(args.preprocessed_dir),
        )
        if not args.no_render:
            render_all_images(model, preprocessed_dir=str(args.preprocessed_dir))


if __name__ == "__main__":
    main()
