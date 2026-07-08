"""Figure generation for data-driven curvature analysis.

Generates publication-quality figures for all 4 phases:
  Phase 1: Data-driven graph structure + curvature results
  Phase 2: Method benchmarking comparison
  Phase 3: Verdict prediction ROC + calibration
  Phase 4: Cross-domain replication
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def fig_degree_curvature(results, output_path, graph_name=""):
    """Degree vs mean curvature scatter for mechanism/hub nodes."""
    dc = results.get("degree_curvature", {})
    if not dc:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_xlabel("Node degree")
    ax.set_ylabel("Mean Ollivier-Ricci curvature")
    ax.set_title(f"Degree confound ({graph_name})\n$r = {dc['r']:.3f}$, $p = {dc['p']:.4f}$")

    ax.text(0.05, 0.95, f"$n = {dc['n_nodes']}$ hub nodes",
            transform=ax.transAxes, va="top", fontsize=9)

    fig.savefig(output_path)
    plt.close(fig)


def fig_edge_z_distribution(results, output_path, graph_name=""):
    """Distribution of edge z-scores from null model."""
    edge_null = results.get("edge_null", {})
    if not edge_null:
        return

    bottleneck = edge_null.get("bottleneck_edges", {})
    redundant = edge_null.get("redundant_edges", {})

    all_z = []
    for edges_dict in [bottleneck, redundant]:
        for v in edges_dict.values():
            all_z.append(v["z"])

    if not all_z:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(all_z, bins=30, color="steelblue", edgecolor="white", alpha=0.7)
    ax.axvline(-1.96, color="red", linestyle="--", linewidth=1, label="$|z| = 1.96$")
    ax.axvline(1.96, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Edge-level curvature $z$-score")
    ax.set_ylabel("Count")
    ax.set_title(f"Edge null distribution ({graph_name})\n"
                 f"{edge_null['n_bottleneck']} bottleneck, {edge_null['n_redundant']} redundant")
    ax.legend()

    fig.savefig(output_path)
    plt.close(fig)


def fig_method_comparison(results, output_path, graph_name=""):
    """Bar chart comparing methods for verdict discrimination."""
    mc = results.get("method_comparison", {})
    if not mc:
        return

    methods = sorted(mc.keys(), key=lambda m: abs(mc[m].get("cohens_d", 0)), reverse=True)
    d_vals = [mc[m]["cohens_d"] for m in methods]
    p_vals = [mc[m]["p"] for m in methods]
    colors = ["forestgreen" if p < 0.05 else "lightgray" for p in p_vals]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(range(len(methods)), d_vals, color=colors, edgecolor="white")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([m.replace("_", " ") for m in methods])
    ax.set_xlabel("Cohen's $d$ (Disconfirmed vs rest)")
    ax.set_title(f"Method comparison ({graph_name})")
    ax.axvline(0, color="black", linewidth=0.5)

    for i, (d, p) in enumerate(zip(d_vals, p_vals)):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.text(d + 0.02 * np.sign(d), i, f"$p={p:.3f}$ {sig}", va="center", fontsize=8)

    fig.savefig(output_path)
    plt.close(fig)


def fig_verdict_prediction_roc(results, output_path, graph_name=""):
    """ROC curve from LOOCV verdict prediction."""
    vp = results.get("verdict_prediction", {})
    if not vp or "predictions" not in vp:
        return

    from sklearn.metrics import roc_curve

    preds = vp["predictions"]
    y_true = [p["true"] for p in preds]
    y_score = [p["pred_prob"] for p in preds]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = vp["auc"]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Verdict prediction LOOCV ({graph_name})\n"
                 f"$n = {vp['n_total']}$, {vp['n_positive']} Disconfirmed")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    fig.savefig(output_path)
    plt.close(fig)


def fig_cross_domain_comparison(domain_results, output_path):
    """Compare curvature findings across disease domains."""
    domains = list(domain_results.keys())
    metrics = {
        "degree_curvature_r": [],
        "n_bottleneck": [],
        "n_redundant": [],
        "n_edges": [],
    }

    for d in domains:
        r = domain_results[d]
        metrics["degree_curvature_r"].append(r.get("degree_curvature", {}).get("r", np.nan))
        metrics["n_bottleneck"].append(r.get("edge_null", {}).get("n_bottleneck", 0))
        metrics["n_redundant"].append(r.get("edge_null", {}).get("n_redundant", 0))
        metrics["n_edges"].append(r.get("metadata", {}).get("n_edges", 0))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Degree-curvature correlation across domains
    ax = axes[0]
    colors = plt.cm.Set2(np.linspace(0, 1, len(domains)))
    ax.barh(range(len(domains)), metrics["degree_curvature_r"], color=colors)
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels([d.capitalize() for d in domains])
    ax.set_xlabel("Degree-curvature $r$")
    ax.set_title("Degree confound by domain")

    # Bottleneck/redundant ratio
    ax = axes[1]
    bottleneck_frac = [b / e if e > 0 else 0
                       for b, e in zip(metrics["n_bottleneck"], metrics["n_edges"])]
    redundant_frac = [r / e if e > 0 else 0
                      for r, e in zip(metrics["n_redundant"], metrics["n_edges"])]
    x = np.arange(len(domains))
    ax.bar(x - 0.2, bottleneck_frac, 0.35, label="Bottleneck", color="indianred")
    ax.bar(x + 0.2, redundant_frac, 0.35, label="Redundant", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in domains], rotation=30)
    ax.set_ylabel("Fraction of edges")
    ax.set_title("Anomalous edges by domain")
    ax.legend()

    # Graph size
    ax = axes[2]
    ax.barh(range(len(domains)), metrics["n_edges"], color=colors)
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels([d.capitalize() for d in domains])
    ax.set_xlabel("Number of edges")
    ax.set_title("Graph size by domain")

    fig.suptitle("Cross-domain replication", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def generate_all_figures(results, output_dir, graph_name=""):
    """Generate all figures for a single graph analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = graph_name.replace(" ", "_").lower() if graph_name else "graph"

    fig_degree_curvature(results, output_dir / f"{prefix}_degree_curvature.pdf", graph_name)
    fig_edge_z_distribution(results, output_dir / f"{prefix}_edge_z_dist.pdf", graph_name)
    fig_method_comparison(results, output_dir / f"{prefix}_method_comparison.pdf", graph_name)
    fig_verdict_prediction_roc(results, output_dir / f"{prefix}_roc.pdf", graph_name)

    print(f"Figures saved to {output_dir}/{prefix}_*.pdf")
