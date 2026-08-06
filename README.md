# Autoencoder-based regularization methods for parametric and inverse projections

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![Uses: venv](https://img.shields.io/badge/Environment-venv-blue)](https://docs.python.org/3/library/venv.html)
[![OSF Project](https://img.shields.io/badge/OSF-View%20Project-lightgrey)](https://osf.io/54byx)
[![ScienceDirect](https://img.shields.io/badge/ScienceDirect-View-orange)](https://www.sciencedirect.com/science/article/pii/S0097849326000233)

📄 **Paper:** [Link](https://frederikdennig.com/publications/Dennig2026Autoencoder)

## Key Features

* Trains autoencoders (AE, VAE [1], VAE with full/isotropic covariance) whose 2D latent spaces align with precomputed projections (t-SNE [2], UMAP [3], PCA, Isomap, LLE, MDS).
* Supports multiple loss components: reconstruction (BCE, MSE), projection alignment (MSE to 2D target), and regularization (KL divergence [1], differential entropy, Jacobian Frobenius norm).
* Evaluates projection quality via trustworthiness and continuity [4] metrics at multiple neighborhood sizes ($k = 2, 4, 8, 16, \ldots$) using Numba-accelerated computation.
* Provides visualization tools: reconstruction comparisons, latent space scatter plots, decoded grids, gradient magnitude maps, and decision boundary maps.
* YAML-based experiment configuration with deterministic seeding, early stopping, and automated parameter sweeps.

## Overview

**Basic Idea:**

![Autoencoder-based regularization for parametric and inverse projections][1]

In our framework, the *encoder* network learns a parametric projection $P$ mapping new data record $x_i$ into 2D space (as $\hat{y}_i$). The *decoder* learns the inverse projection $P^{-1}$ generating a high-dimensional sample $\hat{x}_i$ from any 2D point $y_i$. Our approaches can utilize the losses: $\mathcal{L}_{\text{recon}}$, ensuring correct reconstruction; $\mathcal{L}_{\text{proj}}$, aligning $\hat{y}$ with points of the projection; and $\mathcal{L}_{\text{reg}}$, regularizing the latent space to avoid discontinuity.

![Framework for Autoencoder-based parametric and inverse projections][2]

In our framework (a), the *encoder* network learns a parametric projection $P$ mapping new data record $x_i$ into 2D space (as $\hat{y}_i$). The *decoder* learns the inverse projection $P^{-1}$ generating a high-dimensional sample $\hat{x}_i$ from any 2D point $y_i$. Our approaches can utilize the losses: $\mathcal{L}_{\text{recon}}$, ensuring correct reconstruction; $\mathcal{L}_{\text{proj}}$, aligning $\hat{y}$ with points of the projection; and $\mathcal{L}_{\text{reg}}$, regularizing the latent space to avoid discontinuity. We evaluate three architectures: (b) Two feed-forward NNs, i.e., *Projector and Reconstructor* (P&R), are trained separately to learn a forward and inverse mapping between data and projection spaces without end-to-end optimization or latent space regularization. (c) An *Autoencoder* (AE) is trained end-to-end, with a joint loss that combines reconstruction accuracy and a latent-space alignment term to match a target projection. (d) A *Variational Autoencoder* (VAE) extends the AE by introducing stochastic latent representations, enforcing structure through a KL divergence term and aligning the latent space to a target projection using either sampled or mean-based loss.

[1]: images/overview.png
[2]: images/framework.png

## Requirements

* **Python** >= 3.12 ([Python 3.12.x](https://www.python.org/downloads/release/python-3120/))
* **Virtual environment**: [venv](https://docs.python.org/3/library/venv.html)
* **CUDA** (optional): GPU acceleration for training

## How to Run

### 1. Setup Environment

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Install dev dependencies (Ruff, Black, isort)
pip install -r requirements-dev.txt
```

### 2. Run Experiments

```bash
# Train models from YAML configs (defaults: ./configs, ./records, ./models, ./preprocessed)
python3 trainer.py

# Or point each I/O directory explicitly (e.g. to consume the experiment bundle)
python3 trainer.py \
  --config-dir experiments/table-data/models \
  --records-dir experiments/table-data/records \
  --models-dir experiments/table-data/models \
  --preprocessed-dir experiments/preprocessed
```

This will:
- Load every YAML experiment config from `--config-dir` (default `./configs/`)
- Load datasets (MNIST, FashionMNIST, KMNIST, COIL-20, HAR) with precomputed 2D projections from `--preprocessed-dir`
- Train autoencoder models to align latent spaces with target projections
- Evaluate trustworthiness/continuity on the test set
- Save model checkpoints, training logs, and metric records to `--models-dir` / `--records-dir`

Already-trained configs (a matching `*.test.csv` exists) are skipped; pass `--force`
to retrain, or `--no-render` to skip evaluation-image generation.

### 3. Generate Results Tables

By default the scripts read records/configs from `./records` and `./models`:

```bash
# Aggregate metrics and generate the main results table (terminal, or --latex)
python3 results_table.py
python3 results_table.py --latex

# Parameter-sweep data tables
python3 sweep_tables.py --sweep omega
python3 sweep_tables.py --sweep alphabeta
```

#### Reproducing the paper

The full experiment artifacts (configs, trained checkpoints, records, projection
targets) live in `experiments/` — too large for the repo (gitignored). Download
the experiment data from OSF (<https://osf.io/tb958>) and unpack it here. Then
point the scripts at the relevant experiment directory:

```bash
# Main results table (Table 3)
python3 results_table.py --latex \
  --records-dir experiments/table-data/records \
  --models-dir experiments/table-data/models

# Sweep figures' data
python3 sweep_tables.py --sweep omega \
  --records-dir experiments/parameter-sweep/records \
  --models-dir experiments/parameter-sweep/models
python3 sweep_tables.py --sweep alphabeta \
  --records-dir experiments/parameter-sweep/records \
  --models-dir experiments/parameter-sweep/models
```

### 4. Lint and Format

```bash
./dev.sh lint    # Ruff check
./dev.sh format  # Black + isort
./dev.sh fix     # Ruff format + Black + isort
```

## File Overview

| File | Description |
| --- | --- |
| `trainer.py` | Main training pipeline: config loading, training loop, evaluation, checkpointing. |
| `config.py` | Dataclass-based YAML configuration system for model architecture, losses, and training. |
| `models.py` | Model architectures: AE, VAE (diagonal), VAEFull (full covariance), VAEIsotropic, ProjectorReconstructor. |
| `loss_function.py` | Loss function composition: reconstruction, projection, and regularization terms. |
| `loss_function_impl.py` | JIT-compiled loss implementations: KL divergence [1], differential entropy, Jacobian Frobenius. |
| `data_loader.py` | Dataset loading (MNIST, FashionMNIST, KMNIST, COIL-20, HAR) with paired 2D projections. |
| `metrics.py` | Numba-optimized trustworthiness and continuity [4] computation. |
| `projections.py` | Precomputes 2D projections (t-SNE [2], UMAP [3], PCA, MDS, Isomap, LLE) for datasets. |
| `visual_eval.py` | Visualization driver (`render_all_images`): reconstruction comparisons, latent scatter plots, decoded grids, gradient and decision maps. |
| `gradient_map.py` | Computes and renders decoder gradient magnitude heatmaps over the latent space. |
| `decision_map.py` | Generates classifier decision boundary maps over the latent space. |
| `results_table.py` | Aggregates test records into the main results table (parametric/inverse projection, trustworthiness, continuity, epochs, time) as terminal or LaTeX output; also regenerates `aggregated_trust_continuity.csv`. |
| `sweep_tables.py` | Builds the parameter-sweep data tables (ω for the AE; α/β grids for the VAE). |
| `utils.py` | Utility functions: deterministic seeding. |
| `dev.sh` | Shell helper for linting and formatting (Ruff, Black, isort). |

## Key Directories

| Directory | Description |
| --- | --- |
| `configs/` | YAML experiment configs consumed by `trainer.py`. |
| `models/` | Saved model checkpoints (`.pt`) and their configs (`.yaml`). |
| `records/` | Training logs (`.train.csv`, `.val.csv`, `.test.csv`, `.truts-cont.csv`). |
| `preprocessed/` | Precomputed 2D projection targets per dataset (t-SNE [2], UMAP [3], PCA, etc.). |
| `datasets/` | Dataset files: HAR, blobs, and rings are tracked; MNIST/FashionMNIST/KMNIST/COIL-20 are downloaded on first use. |
| `images/` | Generated visualization outputs. |
| `experiments/` | Full experiment bundle (configs, checkpoints, records, projection targets); not in the repo — download from [OSF](https://osf.io/tb958). |

## Metrics

### Projection Quality

- **Trustworthiness** [4]: Penalizes false neighbors — points that appear close in the 2D projection but are distant in the original high-dimensional space.
- **Continuity** [4]: Penalizes missing neighbors — points that are close in the original space but distant in the projection.

Both metrics are computed at neighborhood sizes $k = 2, 4, 8, 16, \ldots$ using Numba-parallelized routines.

### Loss Components

- **Reconstruction loss**: BCE or MSE between input and decoder output.
- **Projection loss**: MSE between the encoder's 2D latent representation and the precomputed projection target.
- **Regularization loss**: KL divergence [1] (diagonal or full covariance), differential entropy, or Jacobian Frobenius norm.

## Loading a Trained Model

`load_model` reads `./models/{name}.pt` + `./models/{name}.yaml`, so the checkpoint
must exist there first — either from your own training run or copied from the
downloaded experiment bundle.

```python
from models import load_model

model = load_model("fmnist_tsne_vae_sweep-777_0aeb65d7")
```

## Datasets

The HAR (Human Activity Recognition Using Smartphones) dataset is redistributed here under
CC BY 4.0 from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) [5].

## References

[1] Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. *ICLR*.

[2] van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. *Journal of Machine Learning Research*, 9(86), 2579–2605.

[3] McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv:1802.03426*.

[4] Venna, J., & Kaski, S. (2001). Neighborhood Preservation in Nonlinear Projection Methods: An Experimental Study. *ICANN*, 485–491.

[5] Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). A Public Domain Dataset for Human Activity Recognition Using Smartphones. *ESANN 2013*, 437–442.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
