import io
import re
import zipfile
from functools import partial
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ProjectedLabeledDataset(Dataset):
    def __init__(self, indices, vectors, points_2d, labels):
        self.indices = indices
        self.vectors = vectors
        self.points_2d = points_2d
        self.labels = labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return self.vectors[i], self.points_2d[i], self.labels[i]


def load_image_dataset(dataset_class, root: str = "./datasets") -> Tuple[torch.Tensor, torch.Tensor]:
    transform = transforms.ToTensor()
    train = dataset_class(root=root, train=True, download=True, transform=transform)
    test = dataset_class(root=root, train=False, download=True, transform=transform)
    vectors = torch.cat([train.data, test.data], dim=0).float().div(255).view(-1, 28 * 28)
    labels = torch.cat([train.targets, test.targets], dim=0)
    return vectors, labels


def load_har(path: str = "./datasets/HAR") -> Tuple[torch.Tensor, torch.Tensor]:
    train = pd.read_csv(f"{path}/train.csv")
    test = pd.read_csv(f"{path}/test.csv")
    df = pd.concat([train, test], ignore_index=True)
    labels = pd.factorize(df["Activity"])[0]
    df.drop(["subject", "Activity"], axis=1, inplace=True)
    vectors = torch.tensor(df.values, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    return vectors, labels


def load_blobs(path: str = "./datasets") -> Tuple[torch.Tensor, torch.Tensor]:
    df = pd.read_csv(f"{path}/blobs.csv")
    labels = pd.factorize(df["labels"])[0]
    df.drop(["labels"], axis=1, inplace=True)
    vectors = torch.tensor(df.values, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    return vectors, labels


def load_rings(path: str = "./datasets") -> Tuple[torch.Tensor, torch.Tensor]:
    df = pd.read_csv(f"{path}/interlacing_rings.csv")
    labels = pd.factorize(df["labels"])[0]
    df.drop(["id", "labels"], axis=1, inplace=True)
    vectors = torch.tensor(df.values, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    return vectors, labels


# --- Configuration ---
DATA_DIR = Path("./datasets/coil20")
ZIP_URL = "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip"
IMAGE_SIZE = (64, 64)  # size to flatten to


def download_and_extract(url: str, extract_to: Path):
    extract_to.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    print(f"Downloading {url} ...")
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        print(f"Extracting to {extract_to} ...")
        z.extractall(path=extract_to)
    print("Download and extraction complete.")


def load_coil20(image_size=(64, 64)):
    # Download if necessary
    if not (DATA_DIR.exists() and any(DATA_DIR.iterdir())):
        download_and_extract(ZIP_URL, DATA_DIR)
    else:
        print(f"Using existing files in {DATA_DIR}")

    # Try common subdirectories created by the zip
    candidates = [DATA_DIR, DATA_DIR / "coil-20", DATA_DIR / "coil-20-unproc", DATA_DIR / "coil-20_unproc"]
    data_folder = next((c for c in candidates if c.exists() and any(c.rglob("*.png"))), DATA_DIR)

    # Find all image files
    patterns = ["**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.bmp", "**/*.pgm"]
    files = []
    for p in patterns:
        files.extend(sorted(data_folder.glob(p)))
    if not files:
        raise RuntimeError(f"No image files found in {data_folder}")

    regex = re.compile(r"obj0*(\d+)")
    vectors = []
    labels = []

    for f in files:
        try:
            img = Image.open(f).convert("L")
            img = img.resize(image_size, Image.BILINEAR)  # type: ignore
            arr = np.asarray(img, dtype=np.float32).ravel() / 255.0  # normalize to [0,1]
            vectors.append(arr)

            m = regex.search(f.name)
            label = int(m.group(1)) if m else 0
            labels.append(label)
        except Exception as e:
            print(f"Warning: failed to load {f}: {e}")

    X = torch.tensor(np.stack(vectors), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    print(f"Loaded {X.shape[0]} images of size {image_size}.")
    return X, y


def load_user_csv(path: str, label_col: int | None = None) -> Tuple[str, torch.Tensor, torch.Tensor]:
    df = pd.read_csv(path)
    if label_col is not None:
        labels = pd.factorize(df[label_col])[0]
        df = df.drop(columns=[label_col])
    else:
        labels = np.zeros(len(df), dtype=np.int64)

    num_dims = df.shape[1]
    name = f"{Path(path).stem}_{num_dims}"
    save_path = Path("./datasets")
    save_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path / f"{name}.csv", index=False)

    vectors = torch.tensor(df.values, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    return name, vectors, labels


def load_csv_to_tensors(csv_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    df = pd.read_csv(csv_path)
    points_2d = torch.tensor(df[["x", "y"]].values, dtype=torch.float32)
    labels = torch.tensor(df["label"].values, dtype=torch.long)
    return points_2d, labels


def create_data_loaders(
    vectors: torch.Tensor,
    points_2d: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 64,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset_size = len(labels)
    val_size = int(val_frac * dataset_size)
    test_size = int(test_frac * dataset_size)
    train_size = dataset_size - val_size - test_size

    indices = torch.randperm(dataset_size).tolist()
    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]

    train_dataset = ProjectedLabeledDataset(train_idx, vectors, points_2d, labels)
    val_dataset = ProjectedLabeledDataset(val_idx, vectors, points_2d, labels)
    test_dataset = ProjectedLabeledDataset(test_idx, vectors, points_2d, labels)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )


def create_loaders_for_dataset(
    dataset_name: str, projection: str, batch_size: int = 64, preprocessed_dir: str = "./preprocessed"
):
    dataset_loaders = {
        "mnist": partial(load_image_dataset, datasets.MNIST),
        "fmnist": partial(load_image_dataset, datasets.FashionMNIST),
        "kmnist": partial(load_image_dataset, datasets.KMNIST),
        "har": load_har,
        "rings": load_rings,
        "blobs": load_blobs,
        "coil20": load_coil20,
    }

    if dataset_name in dataset_loaders:
        vectors, _ = dataset_loaders[dataset_name]()
    else:
        vectors = torch.tensor(pd.read_csv(f"./datasets/{dataset_name}.csv").values, dtype=torch.float32)

    points_2d, labels = load_csv_to_tensors(f"{preprocessed_dir}/{dataset_name}/{projection}.csv")
    return create_data_loaders(vectors, points_2d, labels, batch_size)


def safe_cat(tensors):
    return torch.cat(tensors, dim=0) if tensors else None


def model_outputs(model, loader):
    model.to(device)
    model.eval()

    originals, proj_targets, labels = [], [], []
    reconstructions, latent_means, latent_covariances, latents = [], [], [], []

    with torch.no_grad():
        for original, proj_target, label in loader:
            original = original.to(device)
            proj_target = proj_target.to(device)
            reconstruction, latent, latent_mean, latent_covar = model(original, proj_target)

            originals.append(original.cpu())
            proj_targets.append(proj_target.cpu())
            labels.append(label)
            reconstructions.append(reconstruction.cpu())
            latents.append(latent.cpu())

            if latent_mean is not None:
                latent_means.append(latent_mean.cpu())
            if latent_covar is not None:
                latent_covariances.append(latent_covar.cpu())

    return (
        torch.cat(originals, dim=0),
        torch.cat(proj_targets, dim=0),
        torch.cat(labels, dim=0),
        torch.cat(reconstructions, dim=0),
        torch.cat(latents, dim=0),
        safe_cat(latent_means),
        safe_cat(latent_covariances),
    )
