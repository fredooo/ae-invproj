"""Generate the main results table (paper Table 3).

Aggregates per-run test records into a table with six sections
(parametric-projection MSE, inverse-projection reconstruction error,
trustworthiness, continuity, training epochs, training time) across the four
model variants (P&R, AE, VAE-y_hat, VAE-mu).

Loss metrics (Proj/Recon/Epochs/Train) are averaged over runs from
``records/{base}.test.csv``. Trustworthiness/continuity are averaged over runs
per k, then reduced to mean +/- std over k in {2, 4, ..., n/2} from
``records/{base}.truts-cont.csv`` -- matching the paper.

Also (re)writes the intermediate ``aggregated_trust_continuity.csv``
(per dataset/projection/model/k mean+/-std over runs).

Usage:
    python results_table.py            # pretty terminal table
    python results_table.py --latex    # LaTeX table for the paper
    python results_table.py --max-seed 777   # widen run selection (e.g. for sweeps)
"""

import argparse
from pathlib import Path
from statistics import mean, stdev

import pandas as pd
import yaml

RECORDS = Path("./records")
MODELS = Path("./models")
AGG_CSV = "aggregated_trust_continuity.csv"

# (dataset_id, projection_id) rows in display order. dataset_id/projection_id
# match the strings stored in the config YAML (and hence the record filenames).
ROWS = [
    ("rings", "mds"),
    ("blobs", "umap"),
    ("mnist", "umap"),
    ("kmnist", "umap"),
    ("har", "tsne"),
    ("fmnist", "tsne"),
    ("coil20", "tsne"),
]

DATASET_NAMES = {
    "rings": "Rings",
    "blobs": "Blobs",
    "mnist": "MNIST",
    "kmnist": "KMNIST",
    "har": "HAR",
    "fmnist": "Fashion-MNIST",
    "coil20": "COIL-20",
}
PROJECTION_NAMES = {"mds": "MDS", "umap": "UMAP", "tsne": "t-SNE", "pca": "PCA"}

# Column label -> (model_type, projection-loss target).
COLUMNS = [
    ("P&R", ("pr", "latent")),
    ("AE", ("ae", "latent")),
    ("VAE-y", ("vae", "latent")),
    ("VAE-mu", ("vae", "latent_mean")),
]
COLUMN_LATEX = {
    "P&R": "\\textbf{P\\&R}",
    "AE": "\\textbf{AE}",
    "VAE-y": "\\textbf{VAE-$\\hat{y}$}",
    "VAE-mu": "\\textbf{VAE-$\\mu$}",
}

# (section title, metric key, decimal places). Metric keys map to columns in the
# test CSV, except Trust/Continuity which come from the truts-cont CSV.
SECTIONS = [
    ("Average MSE of the Parametric Projection (lower is better)", "Proj", 3),
    ("Average Reconstruction Error of the Inverse Projection (lower is better)", "Recon", 3),
    ("Average Trustworthiness T(k) with k in {2, 4, 8, ..., n/2} (higher is better)", "Trust", 4),
    ("Average Continuity C(k) with k in {2, 4, 8, ..., n/2} (higher is better)", "Continuity", 4),
    ("Average Number of Training Epochs (lower is better)", "Epochs", 3),
    ("Average Training Time in Seconds (lower is better)", "Train", 3),
]


def _find_yaml(base: str) -> Path | None:
    """Locate the config YAML for a record base name (searches subdirs too)."""
    top = MODELS / f"{base}.yaml"
    if top.exists():
        return top
    matches = list(MODELS.rglob(f"{base}.yaml"))
    return matches[0] if matches else None


def select_runs(dataset, projection, model_type, target, max_seed=9, pr_max_proj_weight=1.0):
    """Return record base names matching one (dataset, projection, model, target) cell.

    Filters mirror the paper's official-run selection: seed <= max_seed, and for
    P&R only the runs whose projection weight is <= pr_max_proj_weight.
    """
    bases = []
    for csv_file in RECORDS.glob(f"{dataset}_{projection}_{model_type}_*.test.csv"):
        base = csv_file.name[: -len(".test.csv")]
        yaml_file = _find_yaml(base)
        if yaml_file is None:
            continue
        with open(yaml_file) as f:
            cfg = yaml.safe_load(f)

        if cfg.get("dataset") != dataset or cfg.get("projection") != projection:
            continue
        if cfg.get("model", {}).get("type") != model_type:
            continue
        if cfg.get("loss_proj", {}).get("target") != target:
            continue

        seed = cfg.get("training", {}).get("seed")
        if seed is None or seed > max_seed:
            continue
        if model_type == "pr" and (cfg.get("loss_proj", {}).get("weight") or 0.0) > pr_max_proj_weight:
            continue

        bases.append(base)

    if len(bases) > 10:
        print(f"Warning: {dataset}/{projection}/{model_type}/{target} matched {len(bases)} runs (>10).")
    return bases


def _agg(values):
    if not values:
        return (None, None)
    if len(values) == 1:
        return (values[0], 0.0)
    return (mean(values), stdev(values))


def loss_metrics(bases):
    """Mean+/-std over runs of Proj/Recon/Epochs/Train from the test CSVs."""
    cols = ["Proj", "Recon", "Epochs", "Train"]
    collected = {c: [] for c in cols}
    for base in bases:
        df = pd.read_csv(RECORDS / f"{base}.test.csv")
        row = df.iloc[0]
        for c in cols:
            collected[c].append(float(row[c]))
    return {c: _agg(v) for c, v in collected.items()}


def trustcont_metrics(bases):
    """Aggregate T(k)/C(k) over runs per k, then take mean and std over
    k in {2, 4, ..., n/2} (paper convention). I.e. average the runs at each k to
    get one curve, then report mean +/- sample std across the k values."""
    trust_by_k, cont_by_k = {}, {}
    for base in bases:
        path = RECORDS / f"{base}.truts-cont.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            k = int(row["k"])
            trust_by_k.setdefault(k, []).append(float(row["Trust"]))
            cont_by_k.setdefault(k, []).append(float(row["Continuity"]))

    def agg(by_k):
        if not by_k:
            return (None, None)
        per_k = [mean(by_k[k]) for k in sorted(by_k)]  # run-averaged value at each k
        return (mean(per_k), stdev(per_k) if len(per_k) > 1 else 0.0)

    return {"Trust": agg(trust_by_k), "Continuity": agg(cont_by_k)}


def collect(max_seed=9):
    """Build results[(dataset, projection)][column_label] = {metric: (mean, std)}."""
    results = {}
    for dataset, projection in ROWS:
        cell = {}
        for col_label, (model_type, target) in COLUMNS:
            bases = select_runs(dataset, projection, model_type, target, max_seed=max_seed)
            metrics = {}
            metrics.update(loss_metrics(bases))
            metrics.update(trustcont_metrics(bases))
            cell[col_label] = metrics
        results[(dataset, projection)] = cell
    return results


def write_aggregated_trustcont(path=AGG_CSV):
    """Regenerate the per-(dataset, projection, model, k) trust/continuity CSV."""
    records = []
    for csv_file in sorted(RECORDS.glob("*.truts-cont.csv")):
        base = csv_file.name[: -len(".truts-cont.csv")]
        yaml_file = _find_yaml(base)
        if yaml_file is None:
            continue
        with open(yaml_file) as f:
            cfg = yaml.safe_load(f)

        dataset = cfg.get("dataset")
        projection = cfg.get("projection")
        model_type = cfg.get("model", {}).get("type")
        target = cfg.get("loss_proj", {}).get("target")
        seed = cfg.get("training", {}).get("seed")
        if not all([dataset, projection, model_type, target]):
            continue

        df = pd.read_csv(csv_file)
        if df.empty or not {"k", "Trust", "Continuity"}.issubset(df.columns):
            continue

        model = f"{model_type}-{target}"
        for _, row in df.iterrows():
            records.append(
                {
                    "dataset": dataset,
                    "projection": projection,
                    "model": model,
                    "k": int(row["k"]),
                    "trust": float(row["Trust"]),
                    "continuity": float(row["Continuity"]),
                    "seed": seed,
                }
            )

    if not records:
        print("No trust/continuity records found; skipping aggregated CSV.")
        return

    df = pd.DataFrame(records)
    agg = (
        df.groupby(["dataset", "projection", "model", "k"])
        .agg(
            trust_mean=("trust", "mean"),
            trust_std=("trust", "std"),
            continuity_mean=("continuity", "mean"),
            continuity_std=("continuity", "std"),
            n_runs=("trust", "count"),
        )
        .reset_index()
        .sort_values(["dataset", "projection", "model", "k"])
    )
    agg.to_csv(path, index=False)
    print(f"Wrote {path} ({len(agg)} rows).")


def _fmt(value, dp, sep):
    mean_val, std_val = value
    if mean_val is None:
        return "-- (--)"
    return f"{mean_val:.{dp}f} {sep} {std_val:.{dp}f}"


def render_terminal(results):
    out = []
    for title, metric, dp in SECTIONS:
        out.append("")
        out.append(title)
        header = ["Dataset", "Projection"] + [c for c, _ in COLUMNS]
        rows = [header]
        for dataset, projection in ROWS:
            line = [DATASET_NAMES[dataset], PROJECTION_NAMES[projection]]
            for col_label, _ in COLUMNS:
                line.append(_fmt(results[(dataset, projection)][col_label].get(metric, (None, None)), dp, "+/-"))
            rows.append(line)
        widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
        for r in rows:
            out.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r)))
    return "\n".join(out)


def render_latex(results):
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\begin{tabular}{cccccc}",
        "\\textbf{Dataset} & \\textbf{Projection} & " + " & ".join(COLUMN_LATEX[c] for c, _ in COLUMNS) + " \\\\",
        "\\hline",
        "\\hline",
    ]
    for title, metric, dp in SECTIONS:
        lines.append(f"\\multicolumn{{6}}{{c}}{{\\textit{{{title}}}}} \\\\")
        lines.append("\\hline")
        for dataset, projection in ROWS:
            row = [f"\\textbf{{{DATASET_NAMES[dataset]}}}", f"\\textbf{{{PROJECTION_NAMES[projection]}}}"]
            for col_label, _ in COLUMNS:
                row.append(_fmt(results[(dataset, projection)][col_label].get(metric, (None, None)), dp, "$\\pm$"))
            lines.append(" & ".join(row) + " \\\\")
        lines.append("\\hline")
    lines += [
        "\\end{tabular}",
        "\\caption{Aggregated metrics and standard deviation (after $\\pm$) of the parametric and inverse "
        "projections, trustworthiness, and continuity on test data for 10 runs each, along with the average "
        "number of training epochs and training time.}",
        "\\label{tab:experiment-data}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


def main():
    global RECORDS, MODELS

    parser = argparse.ArgumentParser(description="Generate the main results table (paper Table 3).")
    parser.add_argument("--latex", action="store_true", help="Emit a LaTeX table instead of a terminal table.")
    parser.add_argument("--max-seed", type=int, default=9, help="Include runs with seed <= this (default 9).")
    parser.add_argument("--records-dir", type=Path, default=RECORDS, help="Directory of *.test.csv / *.truts-cont.csv.")
    parser.add_argument("--models-dir", type=Path, default=MODELS, help="Directory of config *.yaml (recursive).")
    parser.add_argument("--no-aggregate", action="store_true", help="Skip rewriting aggregated_trust_continuity.csv.")
    args = parser.parse_args()
    RECORDS, MODELS = args.records_dir, args.models_dir

    if not args.no_aggregate:
        write_aggregated_trustcont()

    results = collect(max_seed=args.max_seed)
    print(render_latex(results) if args.latex else render_terminal(results))


if __name__ == "__main__":
    main()
