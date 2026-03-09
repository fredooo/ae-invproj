# Autoencoder-based regularization methods for parametric and inverse projections

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![Uses: venv](https://img.shields.io/badge/Environment-venv-blue)](https://docs.python.org/3/library/venv.html)
[![OSF Project](https://img.shields.io/badge/OSF-View%20Project-lightgrey)](https://osf.io/54byx)

📄 **Paper:** [Paper](https://www.sciencedirect.com/science/article/pii/S0097849326000233)

## Key Features

* Trains autoencoders (AE, VAE [1], VAE with full/isotropic covariance) whose 2D latent spaces align with precomputed projections (t-SNE [2], UMAP [3], PCA, Isomap, LLE, MDS).
* Supports multiple loss components: reconstruction (BCE, MSE), projection alignment (MSE to 2D target), and regularization (KL divergence [1], differential entropy, Jacobian Frobenius norm).
* Evaluates projection quality via trustworthiness and continuity [4] metrics at multiple neighborhood sizes ($k = 2, 4, 8, 16, \ldots$) using Numba-accelerated computation.
* Provides visualization tools: reconstruction comparisons, latent space scatter plots, decoded grids, gradient magnitude maps, and decision boundary maps.
* YAML-based experiment configuration with deterministic seeding, early stopping, and automated parameter sweeps.

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
# Train models from YAML configs
python3 trainer.py
```

This will:
- Load YAML experiment configs from the active config directory
- Load datasets (MNIST, FashionMNIST, KMNIST, COIL-20, HAR) with precomputed 2D projections
- Train autoencoder models to align latent spaces with target projections
- Evaluate trustworthiness/continuity on the test set
- Save model checkpoints, training logs, and metric records

### 3. Generate Results Tables

```bash
# Aggregate metrics and generate LaTeX tables
python3 create_tables.py
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
| `visual_eval.py` | Visualization: reconstruction comparisons, latent scatter plots, gradient and decision maps. |
| `visual_eval_extra.py` | Additional visualizations: decoded grids, UMAP grid decodings, latent 2D plots. |
| `gradient_map.py` | Computes and renders decoder gradient magnitude heatmaps over the latent space. |
| `decision_map.py` | Generates classifier decision boundary maps over the latent space. |
| `create_tables.py` | Aggregates experiment records and generates LaTeX comparison tables. |
| `generate_main_table.py` | Collects per-config test metrics and produces summary tables. |
| `sweep.py` | Reads sweep results and builds parameter-vs-metric tables. |
| `sweep_aggregation.py` | Aggregates trustworthiness/continuity CSVs across sweep runs. |
| `dist_cont_eval.py` | Analyzes test results across datasets, projections, and model types. |
| `test_results.py` | Retrieves and summarizes test set metrics for specific configurations. |
| `utils.py` | Utility functions: deterministic seeding. |
| `dev.sh` | Shell helper for linting and formatting (Ruff, Black, isort). |

## Key Directories

| Directory | Description |
| --- | --- |
| `vae_mu_latent/` | YAML experiment configs for VAE sweeps. |
| `the_big_run_yaml/` | YAML configs for large-scale experiment runs. |
| `af_yaml/` | YAML configs for additional experiment batches. |
| `models/` | Saved model checkpoints (`.pt`) and their configs (`.yaml`). |
| `records/` | Training logs (`.train.csv`, `.val.csv`, `.test.csv`, `.truts-cont.csv`). |
| `preprocessed/` | Precomputed 2D projection targets per dataset (t-SNE [2], UMAP [3], PCA, etc.). |
| `datasets/` | Raw dataset files (MNIST, FashionMNIST, KMNIST, HAR). |
| `images/` | Generated visualization outputs. |

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

```python
from models import load_model

model = load_model("fmnist_tsne_vae_sweep-777_0aeb65d7")
```

## References

[1] Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. *ICLR*.

[2] van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. *Journal of Machine Learning Research*, 9(86), 2579–2605.

[3] McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv:1802.03426*.

[4] Venna, J., & Kaski, S. (2001). Neighborhood Preservation in Nonlinear Projection Methods: An Experimental Study. *ICANN*, 485–491.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).