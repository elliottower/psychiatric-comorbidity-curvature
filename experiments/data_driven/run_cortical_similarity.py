"""Cortical similarity graph: ORC on a graph where edge weights are
spatial correlations of cortical thickness effect maps between disorders.

Data source: Patel et al. 2021, JAMA Psychiatry 78(1):47-63.
Figure 2A cross-disorder cortical thickness correlation matrix (6 disorders).
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
    "internalizing": {"MDD"},
    "psychotic": {"SCZ", "BD"},
    "neurodevelopmental": {"ADHD", "ASD"},
    "compulsive": {"OCD"},
}


def build_cortical_graph(tsv_path, p_threshold=0.05):
    G = nx.Graph()
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            d1 = row["disorder1"].strip()
            d2 = row["disorder2"].strip()
            r = float(row["r_cortical"])
            p = float(row["p"])

            G.add_node(d1)
            G.add_node(d2)

            if p > p_threshold or r <= 0:
                continue

            G.add_edge(d1, d2, weight=r, r_cortical=r, p=p)
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
    print(f"[{ts}] Cortical similarity graph: ORC on ENIGMA cortical thickness correlations")

    G = build_cortical_graph("data/enigma_cortical_similarity.tsv")
    n_possible = G.number_of_nodes() * (G.number_of_nodes() - 1) / 2
    density = G.number_of_edges() / n_possible if n_possible > 0 else 0
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, density={density:.2f}")

    print("\nEdge weights:")
    for u, v, d in sorted(G.edges(data=True), key=lambda x: -x[2]["weight"]):
        et = classify_edge(u, v, CLUSTERS)
        print(f"  {u:10s}-{v:10s}  r={d['r_cortical']:.2f}  [{et}]")

    orc = ollivier_ricci_curvature(G, alpha=0.5)

    within_curv = []
    cross_curv = []
    for (u, v), k in orc.items():
        et = classify_edge(u, v, CLUSTERS)
        if et == "within":
            within_curv.append(k)
        else:
            cross_curv.append(k)

    print(f"\nORC values ({len(orc)} edges):")
    for (u, v), k in sorted(orc.items(), key=lambda x: x[1]):
        et = classify_edge(u, v, CLUSTERS)
        print(f"  {u:10s}-{v:10s}  κ={k:+.4f}  [{et}]")

    print(f"\nWithin-cluster: n={len(within_curv)}, κ={np.mean(within_curv):.4f} ± {np.std(within_curv):.4f}")
    print(f"Cross-cluster:  n={len(cross_curv)}, κ={np.mean(cross_curv):.4f} ± {np.std(cross_curv):.4f}")

    if within_curv and cross_curv:
        U, p_mw = stats.mannwhitneyu(cross_curv, within_curv, alternative="two-sided")
        pooled_std = np.sqrt((np.std(within_curv)**2 + np.std(cross_curv)**2) / 2)
        d = (np.mean(within_curv) - np.mean(cross_curv)) / pooled_std if pooled_std > 0 else 0
        print(f"Mann-Whitney U={U}, p={p_mw:.4f}, Cohen's d={d:.3f}")
    else:
        U, p_mw, d = 0, 1.0, 0.0
        print("WARNING: insufficient within or cross edges for comparison")

    # Node curvature ranking
    node_curv = {}
    for n in G.nodes():
        edges_n = [(n, nb) for nb in G.neighbors(n)]
        curv = [orc.get(e, orc.get((e[1], e[0]), 0)) for e in edges_n]
        node_curv[n] = round(float(np.mean(curv)), 4) if curv else 0

    print(f"\nNode ranking (lowest → highest curvature):")
    for n, k in sorted(node_curv.items(), key=lambda x: x[1]):
        print(f"  {n:10s}  κ={k:+.4f}  deg={G.degree(n)}")

    result = {
        "data_source": "ENIGMA cortical thickness spatial correlations (Patel et al. 2021, JAMA Psychiatry)",
        "graph_type": "cortical similarity (r_cortical)",
        "timestamp": ts,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": round(density, 3),
        "n_within": len(within_curv),
        "n_cross": len(cross_curv),
        "within_mean": round(float(np.mean(within_curv)), 4) if within_curv else None,
        "cross_mean": round(float(np.mean(cross_curv)), 4) if cross_curv else None,
        "within_std": round(float(np.std(within_curv)), 4) if within_curv else None,
        "cross_std": round(float(np.std(cross_curv)), 4) if cross_curv else None,
        "U": float(U),
        "p": round(float(p_mw), 6),
        "cohens_d": round(float(d), 3),
        "node_curvature": dict(sorted(node_curv.items(), key=lambda x: x[1])),
        "edges": {},
    }

    for (u, v), k in sorted(orc.items(), key=lambda x: x[1]):
        et = classify_edge(u, v, CLUSTERS)
        result["edges"][f"{u}-{v}"] = {
            "kappa": round(k, 4),
            "r_cortical": G[u][v]["r_cortical"],
            "type": et,
        }

    out_path = output_dir / f"cortical_similarity_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
