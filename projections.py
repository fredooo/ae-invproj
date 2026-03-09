import gc
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
import pandas as pd
import torch
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, Isomap, LocallyLinearEmbedding

from data_loader import create_data_loaders, load_blobs, load_coil20, load_har, load_image_dataset, load_rings

projection_seed = 777


def project_data(vectors, labels, output_prefix, method: str = "umap"):
    print("Calculating", method.upper())
    model = None
    embedding = None
    if method == "umap":
        model = umap.UMAP(random_state=projection_seed)
        embedding = np.array(model.fit_transform(vectors))
    elif method == "tsne":
        model = TSNE(n_components=2, random_state=projection_seed)
        embedding = model.fit_transform(vectors)
    elif method == "mds":
        model = MDS(n_components=2, random_state=projection_seed, n_jobs=-1)
        embedding = model.fit_transform(vectors)
    elif method == "pca":
        model = PCA(n_components=2)
        embedding = model.fit_transform(vectors)
    elif method == "isomap":
        model = Isomap(n_components=2, n_neighbors=15)
        embedding = model.fit_transform(vectors)
    else:
        model = LocallyLinearEmbedding(n_components=2, n_neighbors=15, random_state=projection_seed)
        embedding = model.fit_transform(vectors)

    # Save coordinates and labels to CSV
    df = pd.DataFrame({"x": embedding[:, 0], "y": embedding[:, 1], "label": labels.numpy()})

    csv_path = Path(f"./preprocessed/{output_prefix}/{method}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved coordinates to: {csv_path}")

    train_loader, val_loader, test_loader = create_data_loaders(vectors, torch.from_numpy(embedding), labels)
    plot_projection_from_loader(
        train_loader, "Train Data", f"./images/projections/{method}/{output_prefix}_train_data.png"
    )
    plot_projection_from_loader(
        val_loader, "Validation Data", f"./images/projections/{method}/{output_prefix}_val_data.png"
    )
    plot_projection_from_loader(
        test_loader, "Test Data", f"./images/projections/{method}/{output_prefix}_test_data.png"
    )

    return model, embedding


def sample_umap_grid_and_inverse(umap_model, output_prefix, grid_size=7, img_shape=(28, 28)):
    # Determine min and max range from the fitted UMAP embeddings
    embedding = umap_model.embedding_
    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()

    # Create evenly spaced grid
    x_vals = np.linspace(x_min, x_max, grid_size)
    y_vals = np.linspace(y_min, y_max, grid_size)
    xv, yv = np.meshgrid(x_vals, y_vals)
    grid_points = np.stack([xv.ravel(), yv.ravel()], axis=1)  # Shape: (grid_size * grid_size, 2)

    npy_path = Path(f"./preprocessed/{output_prefix}/umap_grid_points.npy")
    np.save(npy_path, grid_points)
    print(f"Saved grid positions to: {npy_path}")

    # Inverse transform to image space
    inverses = umap_model.inverse_transform(grid_points)
    inverses = torch.tensor(inverses).reshape(-1, *img_shape)

    # Plot in grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    for i in range(grid_size * grid_size):
        row, col = divmod(i, grid_size)
        ax = axes[grid_size - 1 - row, col]
        ax.imshow(inverses[i], cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    img_path = Path(f"./images/projections/umap/{output_prefix}_umap_inverse.png")
    img_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_path)


def process_har():
    print("Processing HAR")
    vectors, labels = load_har()
    output_prefix = "har"
    project_data(vectors, labels, output_prefix, method="isomap")
    project_data(vectors, labels, output_prefix, method="mds")
    project_data(vectors, labels, output_prefix, method="lle")
    project_data(vectors, labels, output_prefix, method="pca")
    project_data(vectors, labels, output_prefix, method="tsne")
    project_data(vectors, labels, output_prefix, method="umap")
    del vectors, labels
    gc.collect()


def process_rings():
    print("Processing rings")
    vectors, labels = load_rings()
    output_prefix = "rings"
    project_data(vectors, labels, output_prefix, method="isomap")
    project_data(vectors, labels, output_prefix, method="mds")
    project_data(vectors, labels, output_prefix, method="lle")
    project_data(vectors, labels, output_prefix, method="pca")
    project_data(vectors, labels, output_prefix, method="tsne")
    project_data(vectors, labels, output_prefix, method="umap")
    del vectors, labels
    gc.collect()


def process_blobs():
    print("Processing blobs")
    vectors, labels = load_blobs()
    output_prefix = "blobs"
    project_data(vectors, labels, output_prefix, method="lle")
    project_data(vectors, labels, output_prefix, method="pca")
    project_data(vectors, labels, output_prefix, method="tsne")
    project_data(vectors, labels, output_prefix, method="umap")
    del vectors, labels
    gc.collect()


def process_coil20():
    print("Processing coil20")
    vectors, labels = load_coil20()
    output_prefix = "coil20"
    for method in ["lle", "pca", "tsne"]:
        project_data(vectors, labels, output_prefix, method=method)
    umap_model, _ = project_data(vectors, labels, output_prefix, method="umap")
    sample_umap_grid_and_inverse(umap_model, output_prefix, grid_size=7)
    del vectors, labels, umap_model
    gc.collect()


def process_image_dataset(dataset_class, name):
    print(f"Processing {name.upper()}")
    vectors, labels = load_image_dataset(dataset_class)
    output_prefix = name
    for method in ["lle", "pca", "tsne"]:
        project_data(vectors, labels, output_prefix, method=method)
    umap_model, _ = project_data(vectors, labels, output_prefix, method="umap")
    sample_umap_grid_and_inverse(umap_model, output_prefix, grid_size=7)
    del vectors, labels, umap_model
    gc.collect()


def plot_projection_from_loader(data_loader, title, filename):
    all_points_2d = []
    all_labels = []

    # Collect all batches
    for _, points_2d, labels in data_loader:
        all_points_2d.append(points_2d)
        all_labels.append(labels)

    # Concatenate everything into single tensors
    all_points_2d = torch.cat(all_points_2d, dim=0).cpu()
    all_labels = torch.cat(all_labels, dim=0).cpu()

    x = all_points_2d[:, 0]
    y = all_points_2d[:, 1]

    num_classes = len(torch.unique(all_labels))
    cmap = "tab10" if num_classes <= 10 else "tab20" if num_classes <= 20 else "turbo"
    norm = BoundaryNorm(boundaries=np.arange(num_classes + 1) - 0.5, ncolors=num_classes)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x, y, c=all_labels, cmap=cmap, norm=norm, s=5, alpha=0.7)
    plt.colorbar(scatter, ticks=range(num_classes), label="Label")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()

    img_path = Path(filename)
    img_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_path)


if __name__ == "__main__":
    process_coil20()
    # process_blobs()
    # process_rings()
    # process_har()
    # process_image_dataset(datasets.MNIST, "mnist")
    # process_image_dataset(datasets.FashionMNIST, "fmnist")
    # process_image_dataset(datasets.KMNIST, "kmnist")
