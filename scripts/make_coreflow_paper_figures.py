#!/usr/bin/env python3
"""Generate CoReFlow paper figures from local experiment outputs.

The script intentionally exports both figures and the exact CSV data behind each
figure so that paper tables and plots remain auditable.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import anndata as ad
except Exception:  # pragma: no cover
    ad = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs"
SUMMARY = OUTDIR / "experiment_summary_current.csv"

COLORS = {
    "CoReFlow": "#0072B2",
    "CoReFlow-SP": "#E69F00",
    "w/o graph": "#D55E00",
    "w/o delta corr": "#009E73",
    "w/o OT": "#CC79A7",
    "Control": "#999999",
    "Additive": "#56B4E9",
    "Other": "#666666",
}

METRIC_LABELS = {
    "pearson_delta": "Pearson Delta ↑",
    "de_spearman_lfc_sig": "DE Spearman LFC ↑",
    "de_direction_match": "Direction Match ↑",
    "pr_auc": "PR-AUC ↑",
    "roc_auc": "ROC-AUC ↑",
    "mse": "MSE ↓",
    "mae": "MAE ↓",
    "discrimination_score_l2": "Discrimination L2 ↑",
}

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def f(value):
    try:
        if value is None or str(value).strip() == "" or str(value).lower() == "nan":
            return None
        return float(value)
    except Exception:
        return None


def read_csv(path: Path):
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def savefig(fig, path_base: Path):
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def load_summary():
    return read_csv(SUMMARY)


def by_run(rows):
    return {r.get("run_name", ""): r for r in rows}


def run_path(row):
    rel = row.get("relative_run_dir") or ""
    return OUTDIR / rel


def metric(row, key):
    val = f(row.get(key))
    return np.nan if val is None else val


def plot_combo_ablation(rows, figdir: Path):
    idx = by_run(rows)
    specs = [
        ("Control", "00_combosciplex_control_s42_g0"),
        ("Additive", "01_combosciplex_additive_s42_g1"),
        ("CoReFlow", "01_combosciplex_no_spectral_propagation_s42_g1"),
        ("CoReFlow-SP", "02_adaptive_gate_combosciplex_full_s42_g7"),
        ("w/o delta corr", "03_combosciplex_no_delta_corr_s42_g3"),
        ("w/o graph", "02_combosciplex_graph_none_s42_g2"),
    ]
    metrics = ["pearson_delta", "de_spearman_lfc_sig", "de_direction_match", "pr_auc", "roc_auc", "mse"]
    data = []
    for method, run in specs:
        if run not in idx:
            continue
        row = idx[run]
        entry = {"method": method, "run_name": run}
        for m in metrics:
            entry[m] = row.get(m, "")
        data.append(entry)
    write_csv(figdir / "data" / "fig3_combosciplex_ablation.csv", data, ["method", "run_name"] + metrics)

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6))
    x = np.arange(len(data))
    labels = [d["method"] for d in data]
    colors = [COLORS.get(label, COLORS["Other"]) for label in labels]
    for ax, m in zip(axes.flat, metrics):
        vals = [f(d[m]) if f(d[m]) is not None else np.nan for d in data]
        ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.6)
        ax.set_title(METRIC_LABELS[m])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        if m != "mse":
            ax.set_ylim(bottom=0)
    fig.suptitle("ComboSciPlex: graph conditioning matters, explicit propagation is optional", y=1.02, fontsize=11)
    savefig(fig, figdir / "fig3_combosciplex_ablation")


def get_holdout_runs(rows):
    idx = by_run(rows)
    core = {
        0: "04_holdout_no_spectral_propagation_f0_s42_g4",
        1: "02_holdout_no_spectral_propagation_f1_s42_g2",
        2: "05_holdout_no_spectral_propagation_f2_s42_g5",
        3: "00_holdout_no_spectral_propagation_f3_s42_g0",
        4: "01_holdout_no_spectral_propagation_f4_s42_g1",
    }
    graph = {
        0: "00_holdout_graph_none_f0_s42_g0",
        1: "01_holdout_graph_none_f1_s42_g1",
        2: "01_holdout_graph_none_f2_s42_g1",
        3: "02_holdout_graph_none_f3_s42_g2",
        4: "03_holdout_graph_none_f4_s42_g3",
    }
    out = []
    for fold, run in core.items():
        if run in idx:
            out.append(("CoReFlow", fold, idx[run]))
    for fold, run in graph.items():
        if run in idx:
            out.append(("w/o graph", fold, idx[run]))
    return out


def plot_holdout_foldwise(rows, figdir: Path):
    metrics = ["pearson_delta", "mse", "de_direction_match", "de_spearman_lfc_sig"]
    hold = get_holdout_runs(rows)
    data = []
    for method, fold, row in hold:
        entry = {"method": method, "fold": fold, "run_name": row.get("run_name", "")}
        for m in metrics:
            entry[m] = row.get(m, "")
        data.append(entry)
    write_csv(figdir / "data" / "fig4_norman_holdout_foldwise.csv", data, ["method", "fold", "run_name"] + metrics)

    folds = sorted({int(d["fold"]) for d in data})
    methods = ["CoReFlow", "w/o graph"]
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 5.6))
    width = 0.36
    for ax, m in zip(axes.flat, metrics):
        for j, method in enumerate(methods):
            vals = []
            for fold in folds:
                found = [d for d in data if d["method"] == method and int(d["fold"]) == fold]
                vals.append(f(found[0][m]) if found else np.nan)
            ax.bar(np.arange(len(folds)) + (j - 0.5) * width, vals, width, label=method, color=COLORS[method], edgecolor="white", linewidth=0.6)
        ax.set_title(METRIC_LABELS[m])
        ax.set_xticks(np.arange(len(folds)))
        ax.set_xticklabels([f"f{fold}" for fold in folds])
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        if m != "mse":
            ax.set_ylim(bottom=0)
    axes.flat[0].legend(frameon=False, ncol=2)
    fig.suptitle("Norman holdout: CoReFlow improves residual fidelity across folds", y=1.02, fontsize=11)
    savefig(fig, figdir / "fig4_norman_holdout_foldwise")


def family(row):
    name = row.get("run_name", "")
    rel = row.get("relative_run_dir", "")
    if "combosciplex" in rel:
        if "control" in name:
            return "Control"
        if "additive" in name:
            return "Additive"
        if "graph_none" in name:
            return "w/o graph"
        if "no_delta" in name:
            return "w/o delta corr"
        if "no_spectral" in name:
            return "CoReFlow"
        return "CoReFlow-SP"
    if "holdout" in rel:
        if "graph_none" in name:
            return "w/o graph"
        if "no_delta" in name:
            return "w/o delta corr"
        if "no_ot" in name:
            return "w/o OT"
        if "no_spectral" in name:
            return "CoReFlow"
        if "full" in name or "adaptive" in name:
            return "CoReFlow-SP"
    if "core_components" in rel:
        if "graph_none" in name or "no_spectral_embedding" in name:
            return "w/o graph"
        return "CoReFlow-SP"
    return "Other"


def plot_tradeoff(rows, figdir: Path):
    selected = []
    for row in rows:
        if row.get("status") != "complete":
            continue
        if not row.get("pearson_delta"):
            continue
        fam = family(row)
        if fam == "Other":
            continue
        selected.append({
            "method_family": fam,
            "run_name": row.get("run_name", ""),
            "dataset_setting": "ComboSciPlex" if "combosciplex" in row.get("relative_run_dir", "") else "Norman holdout/core",
            "pearson_delta": row.get("pearson_delta", ""),
            "de_spearman_lfc_sig": row.get("de_spearman_lfc_sig", ""),
            "pr_auc": row.get("pr_auc", ""),
            "mse": row.get("mse", ""),
        })
    write_csv(figdir / "data" / "fig5_metric_tradeoff.csv", selected, ["method_family", "dataset_setting", "run_name", "pearson_delta", "de_spearman_lfc_sig", "pr_auc", "mse"])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharex=True)
    for fam in ["CoReFlow", "CoReFlow-SP", "w/o graph", "w/o delta corr", "w/o OT", "Control", "Additive"]:
        xs = [f(d["pearson_delta"]) for d in selected if d["method_family"] == fam]
        ys1 = [f(d["de_spearman_lfc_sig"]) for d in selected if d["method_family"] == fam]
        ys2 = [f(d["pr_auc"]) for d in selected if d["method_family"] == fam]
        xs = np.array([np.nan if v is None else v for v in xs], dtype=float)
        ys1 = np.array([np.nan if v is None else v for v in ys1], dtype=float)
        ys2 = np.array([np.nan if v is None else v for v in ys2], dtype=float)
        if len(xs) == 0:
            continue
        axes[0].scatter(xs, ys1, s=42, alpha=0.78, label=fam, color=COLORS.get(fam, COLORS["Other"]), edgecolors="white", linewidth=0.4)
        axes[1].scatter(xs, ys2, s=42, alpha=0.78, label=fam, color=COLORS.get(fam, COLORS["Other"]), edgecolors="white", linewidth=0.4)
    axes[0].set_xlabel("Pearson Delta ↑")
    axes[0].set_ylabel("DE Spearman LFC ↑")
    axes[1].set_xlabel("Pearson Delta ↑")
    axes[1].set_ylabel("PR-AUC ↑")
    for ax in axes:
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_xlim(left=-0.08)
    axes[1].legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.suptitle("Metric trade-off: residual fidelity and DE-centric metrics are not identical", y=1.03, fontsize=11)
    savefig(fig, figdir / "fig5_metric_tradeoff")


def load_results_for_holdout(rows):
    selected = []
    for method, fold, row in get_holdout_runs(rows):
        path = run_path(row) / "results.csv"
        if not path.exists():
            continue
        for rr in read_csv(path):
            selected.append({
                "method": method,
                "fold": fold,
                "run_name": row.get("run_name", ""),
                "perturbation": rr.get("perturbation", ""),
                "pearson_delta": rr.get("pearson_delta", ""),
                "de_spearman_lfc_sig": rr.get("de_spearman_lfc_sig", ""),
                "pr_auc": rr.get("pr_auc", ""),
                "mse": rr.get("mse", ""),
            })
    return selected


def plot_per_condition_distribution(rows, figdir: Path):
    data = load_results_for_holdout(rows)
    write_csv(figdir / "data" / "fig6_per_condition_distribution.csv", data, ["method", "fold", "run_name", "perturbation", "pearson_delta", "de_spearman_lfc_sig", "pr_auc", "mse"])

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.8))
    metrics = ["pearson_delta", "de_spearman_lfc_sig", "pr_auc"]
    methods = ["CoReFlow", "w/o graph"]
    for ax, m in zip(axes, metrics):
        vals = []
        for method in methods:
            arr = [f(d[m]) for d in data if d["method"] == method and f(d[m]) is not None]
            vals.append(arr)
        bp = ax.boxplot(vals, labels=methods, patch_artist=True, showfliers=False, widths=0.55)
        for patch, method in zip(bp["boxes"], methods):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.75)
        ax.set_title(METRIC_LABELS[m])
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Per-condition variability on Norman holdout", y=1.02, fontsize=11)
    savefig(fig, figdir / "fig6_per_condition_distribution")


def plot_training_curves(rows, figdir: Path):
    curve_rows = []
    for method, fold, row in get_holdout_runs(rows):
        p = run_path(row) / "training_summary.json"
        if not p.exists():
            continue
        js = json.load(open(p))
        hist = js.get("history", [])
        for h in hist:
            curve_rows.append({
                "method": method,
                "fold": fold,
                "run_name": row.get("run_name", ""),
                "step": h.get("step", ""),
                "train_loss": h.get("train_loss", ""),
                "validation_loss": h.get("validation_loss", ""),
                "val_pearson_delta": h.get("val_pearson_delta", ""),
                "lr": h.get("lr", ""),
            })
    write_csv(figdir / "data" / "fig7_training_curves.csv", curve_rows, ["method", "fold", "run_name", "step", "train_loss", "validation_loss", "val_pearson_delta", "lr"])

    grouped = defaultdict(lambda: defaultdict(list))
    for r in curve_rows:
        step = int(r["step"])
        val = f(r["val_pearson_delta"])
        if val is not None:
            grouped[r["method"]][step].append(val)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for method in ["CoReFlow", "w/o graph"]:
        steps = sorted(grouped[method])
        means = [np.mean(grouped[method][s]) for s in steps]
        stds = [np.std(grouped[method][s]) for s in steps]
        ax.plot(steps, means, label=method, color=COLORS[method], linewidth=2)
        ax.fill_between(steps, np.array(means) - np.array(stds), np.array(means) + np.array(stds), color=COLORS[method], alpha=0.14, linewidth=0)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation Pearson Delta ↑")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False)
    ax.set_title("Validation dynamics across holdout folds")
    savefig(fig, figdir / "fig7_training_curves")


def as_dense(x):
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def pca_2d_gpu_or_cpu(x, device="cpu"):
    x = np.asarray(x, dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    if device.startswith("cuda") and torch is not None and torch.cuda.is_available():
        with torch.no_grad():
            xt = torch.tensor(x, device=device, dtype=torch.float32)
            # SVD is robust enough after downsampling.
            _, _, v = torch.linalg.svd(xt, full_matrices=False)
            pts = (xt @ v[:2].T).detach().cpu().numpy()
            return pts, "torch_cuda_svd"
    try:
        from sklearn.decomposition import PCA
        pts = PCA(n_components=2, random_state=42).fit_transform(x)
        return pts, "sklearn_cpu_pca"
    except Exception:
        # fallback random projection
        rng = np.random.default_rng(42)
        proj = rng.normal(size=(x.shape[1], 2)).astype(np.float32)
        return x @ proj, "random_projection"


def plot_pred_real_pca(rows, figdir: Path, device="cpu"):
    if ad is None:
        print("[warn] anndata unavailable; skipping PCA figure")
        return
    idx = by_run(rows)
    run = "00_holdout_no_spectral_propagation_f3_s42_g0"
    if run not in idx:
        print("[warn] representative run missing; skipping PCA figure")
        return
    d = run_path(idx[run])
    pred_p = d / "pred.h5ad"
    real_p = d / "real.h5ad"
    if not pred_p.exists() or not real_p.exists():
        print("[warn] pred/real h5ad missing; skipping PCA figure")
        return
    pred = ad.read_h5ad(pred_p)
    real = ad.read_h5ad(real_p)
    rng = np.random.default_rng(42)
    max_n = 1800
    xp = as_dense(pred.X)
    xr = as_dense(real.X)
    ip = rng.choice(xp.shape[0], size=min(max_n, xp.shape[0]), replace=False)
    ir = rng.choice(xr.shape[0], size=min(max_n, xr.shape[0]), replace=False)
    x = np.vstack([xp[ip], xr[ir]])
    labels = np.array(["predicted"] * len(ip) + ["real"] * len(ir))
    pts, backend = pca_2d_gpu_or_cpu(x, device=device)
    out_rows = []
    for (pc1, pc2), label in zip(pts, labels):
        out_rows.append({"source": label, "pc1": float(pc1), "pc2": float(pc2), "backend": backend, "run_name": run})
    write_csv(figdir / "data" / "fig8_pred_real_pca.csv", out_rows, ["source", "pc1", "pc2", "backend", "run_name"])

    fig, ax = plt.subplots(figsize=(5.0, 4.3))
    for label, color in [("real", "#009E73"), ("predicted", "#0072B2")]:
        mask = labels == label
        ax.scatter(pts[mask, 0], pts[mask, 1], s=5, alpha=0.35, label=label, color=color, linewidths=0)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Predicted vs. real cells in PCA space ({backend})")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18, linewidth=0.5)
    savefig(fig, figdir / "fig8_pred_real_pca")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTDIR / "paper_figures" / "coreflow_20260624")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0 for PCA/SVD figure")
    args = parser.parse_args()
    rows = load_summary()
    figdir = args.output
    figdir.mkdir(parents=True, exist_ok=True)
    (figdir / "data").mkdir(parents=True, exist_ok=True)

    plot_combo_ablation(rows, figdir)
    plot_holdout_foldwise(rows, figdir)
    plot_tradeoff(rows, figdir)
    plot_per_condition_distribution(rows, figdir)
    plot_training_curves(rows, figdir)
    plot_pred_real_pca(rows, figdir, device=args.device)

    manifest = {
        "summary_csv": str(SUMMARY),
        "output_dir": str(figdir),
        "figures": sorted(str(p.name) for p in figdir.glob("*.svg")),
        "data_files": sorted(str(p.name) for p in (figdir / "data").glob("*.csv")),
        "device": args.device,
    }
    with (figdir / "manifest.json").open("w") as fp:
        json.dump(manifest, fp, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
