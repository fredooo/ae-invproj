import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from config import Config
from data_loader import create_loaders_for_dataset, model_outputs
from decision_map import DecisionMap
from gradient_map import GradientMap
from models import EncoderDecoder, load_model
from utils import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mnist_comparison(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    labels: torch.Tensor,
    num_images: int = 10,
    filename: str = "mnist_comparison.png",
):
    # Ensure labels are 1D CPU tensor
    labels = labels.view(-1).cpu()

    # Sort by label (stable sort)
    sort_idx = torch.argsort(labels, stable=True)
    originals = originals[sort_idx]
    reconstructions = reconstructions[sort_idx]
    labels = labels[sort_idx]

    # Select one image per label (in label order)
    selected_indices = []
    seen = set()
    for i, lbl in enumerate(labels.tolist()):
        if lbl not in seen:
            selected_indices.append(i)
            seen.add(lbl)
        if len(selected_indices) == num_images:
            break

    originals = originals[selected_indices]
    reconstructions = reconstructions[selected_indices]
    labels = labels[selected_indices]

    # Unflatten if needed
    if originals.dim() == 2:
        originals = originals.view(-1, 1, 28, 28)
    if reconstructions.dim() == 2:
        reconstructions = reconstructions.view(-1, 1, 28, 28)

    # Clamp to [0,1]
    originals = originals.clamp(0, 1)
    reconstructions = reconstructions.clamp(0, 1)

    # Plot
    fig, axes = plt.subplots(2, len(selected_indices), figsize=(len(selected_indices) * 1.2, 3))

    for i in range(len(selected_indices)):
        # Original
        ax = axes[0, i]
        ax.imshow(originals[i].squeeze().cpu().numpy(), cmap="gray")
        ax.set_title(f"{labels[i]}", fontsize=10)
        ax.axis("off")
        if i == 0:
            ax.set_ylabel("Original", fontsize=12)

        # Reconstruction
        ax = axes[1, i]
        ax.imshow(reconstructions[i].squeeze().cpu().numpy(), cmap="gray")
        ax.axis("off")
        if i == 0:
            ax.set_ylabel("Reconstruction", fontsize=12)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)


def coil20_comparison(
    originals: torch.Tensor, reconstructions: torch.Tensor, num_images: int = 10, filename="coil20_comparison.png"
):
    # Select first N
    originals = originals[:num_images]
    reconstructions = reconstructions[:num_images]

    # Unflatten to 64x64
    originals = originals.view(-1, 1, 64, 64)
    reconstructions = reconstructions.view(-1, 1, 64, 64)

    # Clamp to [0,1] in case outputs aren't already in that range
    originals = originals.clamp(0, 1)
    reconstructions = reconstructions.clamp(0, 1)

    # Plot
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 1.2, 3))

    for i in range(num_images):
        # Original (top row)
        ax = axes[0, i]
        img = originals[i].squeeze().cpu().numpy()  # shape (64, 64)
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        if i == 0:
            ax.set_ylabel("Original", fontsize=12)

        # Reconstruction (bottom row)
        ax = axes[1, i]
        img = reconstructions[i].squeeze().cpu().numpy()
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        if i == 0:
            ax.set_ylabel("Reconstruction", fontsize=12)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)


def scatter_compare(proj_targets, latents, labels, title_left="Projection Targets", title_right="Latents"):
    # Convert to numpy if tensors
    if hasattr(proj_targets, "cpu"):
        proj_targets = proj_targets.cpu().numpy()
    if hasattr(latents, "cpu"):
        latents = latents.cpu().numpy()
    if hasattr(labels, "cpu"):
        labels = labels.cpu().numpy()

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter left: proj_targets
    axes[0].scatter(proj_targets[:, 0], proj_targets[:, 1], c=labels, cmap="tab10", s=10, alpha=0.7)
    axes[0].set_title(title_left)
    axes[0].grid(False)

    # Scatter right: latents
    axes[1].scatter(latents[:, 0], latents[:, 1], c=labels, cmap="tab10", s=10, alpha=0.7)
    axes[1].set_title(title_right)
    axes[1].grid(False)

    # Colorbar for both plots (using the first)
    # cbar = fig.colorbar(sc1, ax=axes.ravel().tolist(), ticks=range(10))
    # cbar.set_label('Class Labels')

    plt.tight_layout()
    plt.savefig("./images/scatterplots.png")


def plot_latent_2d(mu, label, filename, limits=None, size=5, alpha=1.0):
    plt.figure(figsize=(8, 8))  # Square plot
    sizes = torch.ones(mu.size(0)) * size

    plt.scatter(mu[:, 0], mu[:, 1], c=label, s=sizes, cmap="tab10", alpha=alpha)

    # Infer axis limits if not provided
    if limits is None:
        xmin = mu[:, 0].min().item()
        xmax = mu[:, 0].max().item()
        ymin = mu[:, 1].min().item()
        ymax = mu[:, 1].max().item()
    else:
        xmin, xmax, ymin, ymax = limits

    # Apply axis settings
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.axis("equal")
    plt.axis("off")
    plt.grid(False)

    # Save plot
    img_path = Path(filename)
    img_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return (xmin, xmax, ymin, ymax)


def plot_latent_2d_with_grid_from_dir(dir_path, filename, size=5, alpha=1.0):
    dir_path = Path(dir_path)

    # Load umap.csv -> expects columns x, y, label
    df = pd.read_csv(dir_path / "umap.csv")
    mu = torch.tensor(df[["x", "y"]].values, dtype=torch.float32)
    label = df["label"].values

    # Load grid points
    grid = np.load(dir_path / "umap_grid_points.npy")
    grid = grid.reshape(7, 7, 2)
    grid = grid[1:-1, 1:-1, :]
    grid = grid.reshape(-1, 2)

    # Plot
    plt.figure(figsize=(8, 8))
    sizes = torch.ones(mu.size(0)) * size
    plt.scatter(mu[:, 0], mu[:, 1], c=label, s=sizes, cmap="tab10", alpha=alpha)
    plt.scatter(grid[:, 0], grid[:, 1], marker="x", color="black", s=40, linewidths=1.5)

    # Axis settings
    xmin = mu[:, 0].min().item()
    xmax = mu[:, 0].max().item()
    ymin = mu[:, 1].min().item()
    ymax = mu[:, 1].max().item()
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.axis("equal")
    plt.axis("off")
    plt.grid(False)

    # Save plot in the same directory
    output_path = dir_path / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def plot_decoded_grid(ae_model: EncoderDecoder, limits, grid_size=7):
    ae_model.eval()

    xmin, xmax, ymin, ymax = limits

    # Create 2D grid of latent coordinates
    x_lin = np.linspace(xmin, xmax, grid_size)
    y_lin = np.linspace(ymin, ymax, grid_size)
    xx, yy = np.meshgrid(x_lin, y_lin)
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1)  # shape: (grid_size^2, 2)

    # Decode latent vectors
    z = torch.tensor(coords, dtype=torch.float32).to(device)
    with torch.no_grad():
        decoded = ae_model.decoder(z).cpu()

    # Reshape decoded outputs (assuming MNIST-sized images: 28x28)
    decoded = decoded.view(-1, 28, 28)

    # Plot image grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    for idx in range(grid_size * grid_size):
        r, c = divmod(idx, grid_size)
        ax = axes[grid_size - 1 - r, c]  # Flip vertical axis so low y is at bottom
        ax.imshow(decoded[idx], cmap="gray")
        ax.axis("off")
    plt.tight_layout()

    # Save and show
    png_path = Path(f"./images/grid/{ae_model.config.create_filename()}.png")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path)


def plot_decoded_umap_grid(ae_model: EncoderDecoder, clf=None, preprocessed_dir: str = "./preprocessed"):
    ae_model.eval()

    coords = np.load(f"{preprocessed_dir}/{ae_model.config.dataset}/umap_grid_points.npy")
    n_points = coords.shape[0]
    grid_size = int(np.sqrt(n_points))

    assert grid_size * grid_size == n_points, "Grid points must form a square grid."

    # Decode
    z = torch.tensor(coords, dtype=torch.float32).to(device)
    with torch.no_grad():
        decoded = ae_model.decoder(z).cpu()

    # Reshape decoded outputs
    decoded = decoded.view(-1, 28, 28)

    # Classify decoded images and print label grid
    pred_labels = None
    if clf is not None:
        decoded_flat = decoded.view(n_points, -1).numpy()
        pred_labels = clf.predict(decoded_flat)

        label_matrix = pred_labels.reshape(grid_size, grid_size)
        label_matrix = np.flip(label_matrix, axis=0)  # match plotted orientation

        print("Predicted label grid:")
        for row in label_matrix:
            print(row.tolist())

    # Plot grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    for idx in range(n_points):
        r, c = divmod(idx, grid_size)
        ax = axes[grid_size - 1 - r, c]
        ax.imshow(decoded[idx], cmap="gray")
        ax.axis("off")

    plt.tight_layout()

    png_path = Path(f"./images/grid/{ae_model.config.create_filename()}.png")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path)
    plt.close(fig)


def render_all_images(model: EncoderDecoder, preprocessed_dir: str = "./preprocessed"):
    config: Config = model.config

    set_seed(config.training.seed)

    # Load dataset
    _, _, test_loader = create_loaders_for_dataset(
        config.dataset, config.projection, config.training.batch_size, preprocessed_dir=preprocessed_dir
    )

    originals, proj_targets, labels, reconstructions, latents, latent_means, latent_covars = model_outputs(
        model, test_loader
    )

    limits = plot_latent_2d(proj_targets, labels, filename=f"./images/latent/test-gt-{config.dataset}.png")
    plot_latent_2d(
        latents if latent_means is None else latent_means,
        labels,
        limits=limits,
        filename=f"./images/latent/{config.create_filename()}.png",
    )

    if config.dataset in ("mnist", "fmnist", "kmnist"):
        mnist_comparison(
            originals=originals,
            reconstructions=reconstructions,
            labels=labels,
            filename=f"./images/reconstructions/{config.create_filename()}.png",
        )

    if config.dataset == "coil20":
        coil20_comparison(
            originals=originals,
            reconstructions=reconstructions,
            filename=f"./images/reconstructions/{config.create_filename()}.png",
        )

    points = latents if latent_means is None else latent_means

    grad_map = GradientMap(grid_size=800, point_size=2.0, show_metrics=True, scale_max=15.27)
    grad_map.plot(model, points.numpy(), file_path=f"./images/gradients/{model.config.create_filename()}.png")

    clf = LogisticRegression(max_iter=500)
    clf.fit(originals, labels)

    deci_map = DecisionMap(grid_size=800, point_size=5.0)
    deci_map.plot(
        model,
        proj_targets.numpy(),
        labels.numpy(),
        clf,
        file_path=f"./images/decisions/{model.config.create_filename()}.png",
    )

    try:
        if config.projection == "umap":
            plot_decoded_umap_grid(model, preprocessed_dir=preprocessed_dir)
        else:
            plot_decoded_grid(model, limits=limits)
    except Exception:
        print("Unable to create grid. Continuing ...")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a model from a specified path and show the latent space visualization."
    )
    parser.add_argument("--model-name", type=str, required=True, help="name of the model")
    args = parser.parse_args()

    model = load_model(args.model_name)
    model.to(device)
    render_all_images(model)
