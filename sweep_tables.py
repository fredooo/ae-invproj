"""Build the data tables behind the parameter-sweep figures.

Two sweeps, both reading projection/reconstruction test loss from
``records/{base}.test.csv`` for the matching config YAMLs:

  omega      The effect of the projection weight (omega) on AE / MNIST / UMAP:
             projection and reconstruction loss per omega.
  alphabeta  The effect of alpha (projection weight) and beta (regularization
             weight) on VAE-mu / KMNIST / UMAP: recon and proj loss grids.
             (The supplementary VAE-y_hat grid is the same with
              --model vae --target latent --dataset fmnist --projection tsne.)

Usage:
    python sweep_tables.py --sweep omega
    python sweep_tables.py --sweep alphabeta
    python sweep_tables.py --sweep alphabeta --dataset fmnist --projection tsne --target latent
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

RECORDS = Path("./records")
DEFAULT_MODELS = Path("./models")


def collect_runs(models_dir, dataset, projection, model_type, target, comment):
    """Return [(proj_weight, reg_weight, proj_loss, recon_loss)] for matching runs.

    proj_loss/recon_loss are None when the run's test record is absent.
    """
    runs = []
    for yaml_file in sorted(models_dir.rglob("*.yaml")):
        with open(yaml_file) as f:
            cfg = yaml.safe_load(f)

        if cfg.get("dataset") != dataset or cfg.get("projection") != projection:
            continue
        if cfg.get("model", {}).get("type") != model_type:
            continue
        if comment is not None and cfg.get("comment") != comment:
            continue
        loss_proj = cfg.get("loss_proj") or {}
        if target is not None and loss_proj.get("target") != target:
            continue

        proj_w = loss_proj.get("weight")
        reg_w = (cfg.get("loss_reg") or {}).get("weight")

        proj_loss = recon_loss = None
        record = RECORDS / f"{yaml_file.stem}.test.csv"
        if record.exists():
            row = pd.read_csv(record).iloc[0]
            proj_loss, recon_loss = float(row["Proj"]), float(row["Recon"])

        runs.append((proj_w, reg_w, proj_loss, recon_loss))
    return runs


def _fmt(value):
    return f"{value:.4f}" if value is not None else "--"


def _mean(values):
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def omega_table(runs):
    """One row per omega (= projection weight): mean Proj and Recon loss."""
    omegas = sorted({proj_w for proj_w, _, _, _ in runs if proj_w is not None})
    header = ["omega", "Proj", "Recon"]
    rows = [header]
    for omega in omegas:
        sel = [(p, r) for pw, _, p, r in runs if pw == omega]
        rows.append([str(omega), _fmt(_mean([p for p, _ in sel])), _fmt(_mean([r for _, r in sel]))])
    return rows


def alpha_beta_grids(runs):
    """Recon and Proj grids indexed by alpha (proj weight) x beta (reg weight)."""
    alphas = sorted({pw for pw, _, _, _ in runs if pw is not None})
    betas = sorted({rw for _, rw, _, _ in runs if rw is not None})

    def grid(metric_idx):
        header = ["alpha \\ beta"] + [str(b) for b in betas]
        out = [header]
        for a in alphas:
            line = [str(a)]
            for b in betas:
                vals = [run[metric_idx] for run in runs if run[0] == a and run[1] == b]
                line.append(_fmt(_mean(vals)))
            out.append(line)
        return out

    return grid(3), grid(2)  # recon grid, proj grid


def print_table(title, rows):
    print(f"\n{title}")
    if len(rows) <= 1:
        print("  (no matching runs)")
        return
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print("  ".join(cell.rjust(widths[i]) for i, cell in enumerate(r)))


def main():
    global RECORDS

    parser = argparse.ArgumentParser(description="Build parameter-sweep data tables.")
    parser.add_argument("--sweep", choices=["omega", "alphabeta"], required=True)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--records-dir", type=Path, default=RECORDS, help="Directory of *.test.csv records.")
    parser.add_argument("--dataset")
    parser.add_argument("--projection")
    parser.add_argument("--model")
    parser.add_argument("--target")
    parser.add_argument("--comment", default="sweep-777")
    args = parser.parse_args()
    RECORDS = args.records_dir

    if args.sweep == "omega":
        dataset = args.dataset or "mnist"
        projection = args.projection or "umap"
        model = args.model or "ae"
        target = args.target or "latent"
        runs = collect_runs(args.models_dir, dataset, projection, model, target, args.comment)
        print(f"omega sweep: {model} / {dataset} / {projection} (comment={args.comment})")
        print_table("Projection and reconstruction loss vs omega", omega_table(runs))
    else:
        dataset = args.dataset or "kmnist"
        projection = args.projection or "umap"
        model = args.model or "vae"
        target = args.target or "latent_mean"
        runs = collect_runs(args.models_dir, dataset, projection, model, target, args.comment)
        print(f"alpha-beta sweep: {model} ({target}) / {dataset} / {projection} (comment={args.comment})")
        recon_grid, proj_grid = alpha_beta_grids(runs)
        print_table("Reconstruction loss (alpha x beta)", recon_grid)
        print_table("Projection loss (alpha x beta)", proj_grid)


if __name__ == "__main__":
    main()
