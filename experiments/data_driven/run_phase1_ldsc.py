"""Phase 1: Run pipeline on LDSC genetic correlation graph.

Builds a weighted disorder-disorder graph from published LDSC rg values
(Lee et al. 2019 CDG2, Grotzinger et al. 2022). Edge weights = |rg|.
This is the data-driven graph that addresses the "artifact of curation" critique.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import stats
from tqdm import tqdm

from weighted_orc import (
    ollivier_ricci_curvature,
    compute_all_edge_features,
)
from figures import generate_all_figures


DISORDER_LABELS = {
    "ADHD": "ADHD",
    "AN": "Anorexia Nervosa",
    "ASD": "Autism",
    "BD": "Bipolar Disorder",
    "MDD": "Major Depression",
    "OCD": "OCD",
    "SCZ": "Schizophrenia",
    "TS": "Tourette Syndrome",
    "ANX": "Anxiety Disorders",
    "PTSD": "PTSD",
    "ALCH": "Alcohol/Substance",
}

INTERNALIZING = {"MDD", "ANX", "PTSD"}
PSYCHOTIC = {"SCZ", "BD"}
NEURODEVELOPMENTAL = {"ADHD", "ASD", "TS"}
COMPULSIVE = {"AN", "OCD"}
SUBSTANCE = {"ALCH"}


def build_ldsc_graph(rg_file, min_abs_rg=0.0, p_threshold=1.0):
    """Build weighted graph from LDSC genetic correlations."""
    import csv
    G = nx.Graph()

    with open(rg_file) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            t1 = row["trait1"].strip()
            t2 = row["trait2"].strip()
            rg = float(row["rg"])
            se = float(row["se"])
            p = float(row["p"])

            if abs(rg) < min_abs_rg or p > p_threshold:
                continue

            G.add_node(t1, label=DISORDER_LABELS.get(t1, t1))
            G.add_node(t2, label=DISORDER_LABELS.get(t2, t2))
            G.add_edge(t1, t2, weight=abs(rg), rg=rg, se=se, p=p,
                       rg_sign=int(np.sign(rg)))

    return G


def classify_edge(u, v):
    """Classify an edge by disorder domain."""
    u_groups = []
    v_groups = []
    for name, group in [("internal", INTERNALIZING), ("psychotic", PSYCHOTIC),
                        ("neurodev", NEURODEVELOPMENTAL), ("compulsive", COMPULSIVE),
                        ("substance", SUBSTANCE)]:
        if u in group:
            u_groups.append(name)
        if v in group:
            v_groups.append(name)

    if u_groups and v_groups and u_groups[0] == v_groups[0]:
        return f"within_{u_groups[0]}"
    return "cross_domain"


def main():
    output_dir = Path("../../results/psych/data_driven")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Build graph — include all significant pairs (p < 0.05)
    print(f"[{ts}] Building LDSC genetic correlation graph...")
    G = build_ldsc_graph("data/ldsc_psychiatric_rg.tsv", min_abs_rg=0.0, p_threshold=0.05)
    print(f"  {G.number_of_nodes()} disorders, {G.number_of_edges()} edges (p < 0.05)")

    # Also build a graph with all pairs for comparison
    G_all = build_ldsc_graph("data/ldsc_psychiatric_rg.tsv", min_abs_rg=0.0, p_threshold=1.0)
    print(f"  {G_all.number_of_nodes()} disorders, {G_all.number_of_edges()} edges (all)")

    results = {
        "metadata": {
            "timestamp": ts,
            "graph_type": "ldsc_genetic_correlations",
            "sources": ["Lee2019_CDG2_Cell", "Grotzinger2022_NatGenet"],
            "n_disorders": G.number_of_nodes(),
            "n_edges_sig": G.number_of_edges(),
            "n_edges_all": G_all.number_of_edges(),
        }
    }

    # Compute ORC on weighted graph
    print(f"\nComputing weighted ORC on significant edges...")
    orc = ollivier_ricci_curvature(G, alpha=0.5)
    results["orc_per_edge"] = {}
    for (u, v), k in sorted(orc.items(), key=lambda x: x[1]):
        edge_type = classify_edge(u, v)
        rg = G[u][v].get("rg", 0)
        results["orc_per_edge"][f"{u}-{v}"] = {
            "kappa": round(k, 4),
            "rg": rg,
            "weight": round(abs(rg), 3),
            "edge_type": edge_type,
        }
        print(f"  {u:5s} -- {v:5s}  κ={k:+.4f}  rg={rg:+.3f}  [{edge_type}]")

    # Degree-curvature correlation
    degrees = [G.degree(n) for n in G.nodes()]
    mean_curv = []
    for n in G.nodes():
        edges = [(n, nb) for nb in G.neighbors(n)]
        curv = [orc.get(e, orc.get((e[1], e[0]), 0)) for e in edges]
        mean_curv.append(np.mean(curv) if curv else 0)

    r, p = stats.pearsonr(degrees, mean_curv)
    results["degree_curvature"] = {"r": round(r, 4), "p": round(p, 6), "n_nodes": len(degrees)}
    print(f"\nDegree-curvature: r={r:.3f}, p={p:.4f}")

    # Weight-permutation null (graph is too dense for degree-preserving rewiring)
    n_perms = 500
    print(f"\nRunning weight-permutation null ({n_perms} permutations)...")
    print("  (Graph is 65% dense — permuting weights on fixed topology)")
    rng = np.random.default_rng(42)
    edges = list(G.edges())
    weights = np.array([G[u][v]["weight"] for u, v in edges])

    null_curvatures = {e: [] for e in edges}
    for i in tqdm(range(n_perms), desc="Weight-perm null"):
        H = G.copy()
        shuffled = rng.permutation(weights)
        for (u, v), w in zip(edges, shuffled):
            H[u][v]["weight"] = w
        kappas = ollivier_ricci_curvature(H, alpha=0.5)
        for e, k in kappas.items():
            null_curvatures[e].append(k)

    bottleneck = {}
    redundant = {}
    edge_null = {}
    for e in edges:
        nulls = np.array(null_curvatures[e])
        mu, sigma = nulls.mean(), nulls.std()
        kappa = orc[e]
        z = (kappa - mu) / sigma if sigma > 1e-10 else 0.0
        edge_null[e] = {"kappa_obs": kappa, "z": z, "mean_null": mu, "std_null": sigma}
        key = f"{e[0]}-{e[1]}"
        if z < -1.96:
            bottleneck[key] = edge_null[e]
        elif z > 1.96:
            redundant[key] = edge_null[e]

    results["edge_null"] = {
        "null_type": "weight_permutation",
        "n_perms": n_perms,
        "n_tested": len(edge_null),
        "n_bottleneck": len(bottleneck),
        "n_redundant": len(redundant),
        "bottleneck_edges": bottleneck,
        "redundant_edges": redundant,
    }
    print(f"  Tested: {len(edge_null)}, Bottleneck: {len(bottleneck)}, Redundant: {len(redundant)}")

    # Cross-domain vs within-domain curvature
    cross_curv = []
    within_curv = []
    for (u, v), k in orc.items():
        et = classify_edge(u, v)
        if et == "cross_domain":
            cross_curv.append(k)
        else:
            within_curv.append(k)

    if cross_curv and within_curv:
        U, p_mw = stats.mannwhitneyu(cross_curv, within_curv, alternative="two-sided")
        results["cross_vs_within"] = {
            "cross_domain_mean": round(np.mean(cross_curv), 4),
            "within_domain_mean": round(np.mean(within_curv), 4),
            "U": float(U),
            "p": round(p_mw, 6),
            "n_cross": len(cross_curv),
            "n_within": len(within_curv),
        }
        print(f"\nCross-domain curvature: {np.mean(cross_curv):.4f} (n={len(cross_curv)})")
        print(f"Within-domain curvature: {np.mean(within_curv):.4f} (n={len(within_curv)})")
        print(f"  Mann-Whitney p={p_mw:.4f}")

    # Which disorders are most central (most negative curvature)?
    node_curv = {}
    for n in G.nodes():
        edges = [(n, nb) for nb in G.neighbors(n)]
        curv = [orc.get(e, orc.get((e[1], e[0]), 0)) for e in edges]
        node_curv[n] = round(np.mean(curv), 4) if curv else 0

    results["node_curvature"] = dict(sorted(node_curv.items(), key=lambda x: x[1]))
    print(f"\nNode curvature ranking (most negative = most bottleneck):")
    for n, k in sorted(node_curv.items(), key=lambda x: x[1]):
        print(f"  {DISORDER_LABELS.get(n, n):25s}  κ_mean={k:+.4f}  deg={G.degree(n)}")

    # Depression still the bottleneck?
    mdd_rank = sorted(node_curv.items(), key=lambda x: x[1])
    mdd_position = [i for i, (n, _) in enumerate(mdd_rank) if n == "MDD"]
    results["depression_rank"] = {
        "position": mdd_position[0] + 1 if mdd_position else None,
        "total": len(mdd_rank),
        "curvature": node_curv.get("MDD", None),
    }

    # Save results
    out_path = output_dir / f"ldsc_weighted_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    # Generate figures
    generate_all_figures(results, output_dir / "figures", "LDSC genetic correlations")


if __name__ == "__main__":
    main()
