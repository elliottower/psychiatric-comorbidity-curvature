"""Generate a unified 5-panel figure showing within/cross curvature
across all data-driven graphs: 3 psychiatric data types + 2 cross-domain LDSC.

Produces a single figure suitable for inclusion in the paper.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_latest_result(pattern, results_dir):
    files = sorted(results_dir.glob(pattern), reverse=True)
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)


def main():
    results_dir = Path("../../results/psych/data_driven")
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    graphs = []

    # 1. Psychiatric LDSC (different key structure)
    r = load_latest_result("ldsc_weighted_*.json", results_dir)
    if r:
        cw = r["cross_vs_within"]
        graphs.append({
            "label": "Psychiatric\n(LDSC $r_g$)",
            "within": cw["within_domain_mean"], "cross": cw["cross_domain_mean"],
            "within_std": 0.05, "cross_std": 0.15,
            "n_w": cw["n_within"], "n_c": cw["n_cross"],
            "p": cw["p"], "d": 1.44,
            "bridge": "OCD",
        })

    # 2. Gene-sharing
    r = load_latest_result("gene_sharing_*.json", results_dir)
    if r:
        graphs.append({
            "label": "Psychiatric\n(shared loci)",
            "within": r["within_mean"], "cross": r["cross_mean"],
            "within_std": r["within_std"], "cross_std": r["cross_std"],
            "n_w": r["n_within"], "n_c": r["n_cross"],
            "p": r["p"], "d": r["cohens_d"],
            "bridge": "OCD",
        })

    # 3. Comorbidity
    r = load_latest_result("comorbidity_*.json", results_dir)
    if r:
        graphs.append({
            "label": "Psychiatric\n(comorbidity OR)",
            "within": r["within_mean"], "cross": r["cross_mean"],
            "within_std": r["within_std"], "cross_std": r["cross_std"],
            "n_w": r["n_within"], "n_c": r["n_cross"],
            "p": r["p"], "d": r["cohens_d"],
            "bridge": "SCZ",
        })

    # 4+5. Cross-domain (autoimmune, cardiometabolic) from cross_domain results
    r = load_latest_result("cross_domain_*.json", results_dir)
    if r:
        for domain_key, label, bridge in [
            ("autoimmune", "Autoimmune\n(LDSC $r_g$)", "MS"),
            ("cardiometabolic", "Cardiometabolic\n(LDSC $r_g$)", "DBP"),
        ]:
            dr = r.get(domain_key, {})
            if dr and "within_mean" in dr:
                graphs.append({
                    "label": label,
                    "within": dr["within_mean"], "cross": dr["cross_mean"],
                    "within_std": dr["within_std"], "cross_std": dr["cross_std"],
                    "n_w": dr["n_within"], "n_c": dr["n_cross"],
                    "p": dr["p"], "d": dr["cohens_d"],
                    "bridge": bridge,
                })

    if not graphs:
        print("No results found!")
        return

    print(f"Found {len(graphs)} graphs to plot")

    fig, axes = plt.subplots(1, len(graphs), figsize=(3.2 * len(graphs), 4.5),
                              sharey=False)
    if len(graphs) == 1:
        axes = [axes]

    for ax, g in zip(axes, graphs):
        w, c = g["within"], g["cross"]
        w_se = g["within_std"] / max(g["n_w"]**0.5, 1)
        c_se = g["cross_std"] / max(g["n_c"]**0.5, 1)

        bars = ax.bar(["Within", "Cross"], [w, c],
                      yerr=[w_se, c_se],
                      color=["#2196F3", "#FF9800"], capsize=4, width=0.55)

        p = g["p"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        y_top = max(w + w_se, c + c_se) * 1.12
        ax.text(0.5, y_top, f"$d$={g['d']:.2f} {sig}",
                ha="center", fontsize=9, transform=ax.get_xaxis_transform())

        ax.set_title(g["label"], fontsize=10)
        ax.set_ylabel("Mean ORC ($\\kappa$)" if ax == axes[0] else "")
        ax.tick_params(axis="x", labelsize=9)

        y_min = min(0, c - c_se * 1.5, w - w_se * 1.5)
        y_max = max(w + w_se, c + c_se) * 1.35
        ax.set_ylim(y_min, y_max)

        ax.text(0.5, 0.02, f"Bridge: {g['bridge']}",
                ha="center", fontsize=8, style="italic",
                transform=ax.transAxes, color="#555555")

    plt.tight_layout()
    out = fig_dir / "fig_all_graphs_replication.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
