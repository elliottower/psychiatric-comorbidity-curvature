"""Gene-sharing graph: ORC on a graph where edge weights are Jaccard
similarity of shared GWAS risk loci between psychiatric disorders.

Data source: Cross-Disorder Group of the PGC (2019), Nature Genetics.
146 pleiotropic loci across 8 psychiatric disorders.

This provides a third independent data type (shared loci counts) alongside
the curated mechanism graph and LDSC genetic correlations.
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import stats
from tqdm import tqdm

from weighted_orc import ollivier_ricci_curvature


CLUSTERS = {
    "psychotic": {"SCZ", "BD"},
    "internalizing": {"MDD", "PTSD"},
    "neurodevelopmental": {"ADHD", "ASD", "TS"},
    "compulsive": {"AN", "OCD"},
}


def build_gene_sharing_graph(tsv_path, min_shared=1):
    G = nx.Graph()
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            d1 = row["disorder1"].strip()
            d2 = row["disorder2"].strip()
            shared = int(row["shared_loci"])
            total_d1 = int(row["total_loci_d1"])
            total_d2 = int(row["total_loci_d2"])

            G.add_node(d1)
            G.add_node(d2)

            if shared < min_shared:
                continue

            jaccard = shared / (total_d1 + total_d2 - shared)
            G.add_edge(d1, d2, weight=jaccard, shared=shared,
                       total_d1=total_d1, total_d2=total_d2)
    return G


def classify_edge(u, v, clusters):
    u_cluster = None
    v_cluster = None
    for name, members in clusters.items():
        if u in members:
            u_cluster = name
        if v in members:
            v_cluster = name
    if u_cluster and v_cluster and u_cluster == v_cluster:
        return "within"
    return "cross"


def weight_permutation_null(G, n_perms=500, alpha=0.5, seed=42):
    rng = np.random.default_rng(seed)
    edges = list(G.edges())
    weights = np.array([G[u][v]["weight"] for u, v in edges])

    null_curvatures = {e: [] for e in edges}
    for _ in tqdm(range(n_perms), desc="Weight-perm null", leave=False):
        H = G.copy()
        shuffled = rng.permutation(weights)
        for (u, v), w in zip(edges, shuffled):
            H[u][v]["weight"] = w
        kappas = ollivier_ricci_curvature(H, alpha=alpha)
        for e, k in kappas.items():
            null_curvatures[e].append(k)

    return null_curvatures


def main():
    output_dir = Path("../../results/psych/data_driven")
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    print(f"[{ts}] Gene-sharing graph: ORC on PGC shared loci")

    G = build_gene_sharing_graph("data/pgc_shared_loci.tsv", min_shared=1)
    n_possible = G.number_of_nodes() * (G.number_of_nodes() - 1) / 2
    density = G.number_of_edges() / n_possible if n_possible > 0 else 0
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, density={density:.2f}")

    print("\nEdge weights (Jaccard similarity of shared loci):")
    for u, v, d in sorted(G.edges(data=True), key=lambda x: -x[2]["weight"]):
        print(f"  {u:5s}-{v:5s}  shared={d['shared']:2d}  Jaccard={d['weight']:.3f}")

    orc = ollivier_ricci_curvature(G, alpha=0.5)

    print(f"\nORC values ({len(orc)} edges):")
    for (u, v), k in sorted(orc.items(), key=lambda x: x[1]):
        et = classify_edge(u, v, CLUSTERS)
        print(f"  {u:5s}-{v:5s}  κ={k:+.4f}  [{et}]")

    within_curv = []
    cross_curv = []
    for (u, v), k in orc.items():
        et = classify_edge(u, v, CLUSTERS)
        if et == "within":
            within_curv.append(k)
        else:
            cross_curv.append(k)

    print(f"\nWithin-cluster: n={len(within_curv)}, κ={np.mean(within_curv):.4f} ± {np.std(within_curv):.4f}")
    print(f"Cross-cluster:  n={len(cross_curv)}, κ={np.mean(cross_curv):.4f} ± {np.std(cross_curv):.4f}")

    if within_curv and cross_curv:
        U, p_mw = stats.mannwhitneyu(cross_curv, within_curv, alternative="two-sided")
        pooled_std = np.sqrt((np.std(within_curv)**2 + np.std(cross_curv)**2) / 2)
        d = (np.mean(within_curv) - np.mean(cross_curv)) / pooled_std if pooled_std > 0 else 0
        print(f"Mann-Whitney U={U}, p={p_mw:.4f}, Cohen's d={d:.3f}")
    else:
        U, p_mw, d = 0, 1.0, 0.0

    # Node curvature ranking
    node_curv = {}
    for n in G.nodes():
        edges_n = [(n, nb) for nb in G.neighbors(n)]
        curv = [orc.get(e, orc.get((e[1], e[0]), 0)) for e in edges_n]
        node_curv[n] = round(float(np.mean(curv)), 4) if curv else 0

    print(f"\nNode ranking (lowest → highest curvature):")
    for n, k in sorted(node_curv.items(), key=lambda x: x[1]):
        print(f"  {n:5s}  κ={k:+.4f}  deg={G.degree(n)}")

    # Weight-permutation null
    print("\nRunning weight-permutation null (500 perms)...")
    null_curv = weight_permutation_null(G, n_perms=500)

    edge_zscores = {}
    for (u, v), k in orc.items():
        null_vals = null_curv.get((u, v), null_curv.get((v, u), []))
        if null_vals:
            z = (k - np.mean(null_vals)) / max(np.std(null_vals), 1e-10)
            edge_zscores[(u, v)] = round(float(z), 3)

    print("\nEdge z-scores (vs weight-permutation null):")
    for (u, v), z in sorted(edge_zscores.items(), key=lambda x: x[1]):
        et = classify_edge(u, v, CLUSTERS)
        print(f"  {u:5s}-{v:5s}  z={z:+.3f}  [{et}]")

    result = {
        "data_source": "PGC Cross-Disorder 2019, shared GWAS loci",
        "graph_type": "gene-sharing (Jaccard of shared loci)",
        "timestamp": ts,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": round(density, 3),
        "n_within": len(within_curv),
        "n_cross": len(cross_curv),
        "within_mean": round(float(np.mean(within_curv)), 4),
        "cross_mean": round(float(np.mean(cross_curv)), 4),
        "within_std": round(float(np.std(within_curv)), 4),
        "cross_std": round(float(np.std(cross_curv)), 4),
        "U": float(U),
        "p": round(float(p_mw), 6),
        "cohens_d": round(float(d), 3),
        "node_curvature": dict(sorted(node_curv.items(), key=lambda x: x[1])),
        "edge_zscores": {f"{u}-{v}": z for (u, v), z in edge_zscores.items()},
        "edges": {},
    }

    for (u, v), k in sorted(orc.items(), key=lambda x: x[1]):
        et = classify_edge(u, v, CLUSTERS)
        result["edges"][f"{u}-{v}"] = {
            "kappa": round(k, 4),
            "shared_loci": G[u][v].get("shared", 0),
            "jaccard": round(G[u][v].get("weight", 0), 4),
            "type": et,
            "z": edge_zscores.get((u, v), edge_zscores.get((v, u), None)),
        }

    out_path = output_dir / f"gene_sharing_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(5, 4))
    n_w, n_c = len(within_curv), len(cross_curv)
    ax.bar(["Within\ncluster", "Cross\ncluster"],
           [np.mean(within_curv), np.mean(cross_curv)],
           yerr=[np.std(within_curv)/max(n_w**0.5, 1), np.std(cross_curv)/max(n_c**0.5, 1)],
           color=["#2196F3", "#FF9800"], capsize=5, width=0.5)
    sig = "***" if p_mw < 0.001 else "**" if p_mw < 0.01 else "*" if p_mw < 0.05 else "n.s."
    y_max = max(np.mean(within_curv) + np.std(within_curv),
                np.mean(cross_curv) + np.std(cross_curv))
    ax.text(0.5, y_max * 1.15, f"p={p_mw:.3f} {sig}", ha="center", fontsize=10)
    ax.set_title(f"Gene-sharing graph\n(9 disorders, {G.number_of_edges()} edges)")
    ax.set_ylabel("Mean ORC (κ)")
    plt.tight_layout()
    fig_path = output_dir / "figures" / "fig_gene_sharing_orc.pdf"
    (output_dir / "figures").mkdir(exist_ok=True)
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
