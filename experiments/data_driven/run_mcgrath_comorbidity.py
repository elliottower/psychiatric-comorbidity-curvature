"""McGrath et al. 2020 comorbidity graph: ORC on a graph where edge weights
are log(HR) from the WHO World Mental Health Survey (N=145,990).

Data source: McGrath et al. (2020) "Age-related changes in the bidirectional
relationship between lifetime mental disorders and subsequent onset of chronic
physical conditions." Table published as supplementary CSV, 24 disorders,
directed hazard ratios. We symmetrize by taking max(HR_ab, HR_ba).

This provides a large, verified epidemiological data source independent of
genetics and expert curation.
"""

import csv
import json
import math
from collections import defaultdict
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
    "mood": {"MDE", "Bipolar disorder", "Dysthymia"},
    "anxiety": {"GAD", "Panic disorder", "Social phobia", "Specific phobia",
                "Agoraphobia", "PTSD", "Adult SAD", "Child SAD"},
    "obsessive": {"OCD"},
    "eating": {"Anorexia nervosa", "Bulimia nervosa", "Binge eating disorder"},
    "externalizing": {"ADHD", "Conduct disorder", "ODD", "IED"},
    "substance": {"Nicotine dependence", "Alcohol abuse", "Alcohol dependence",
                  "Drug abuse", "Drug dependence"},
}


def build_mcgrath_graph(csv_path):
    pair_hrs = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Sex"] != "All" or row["Model"] != "A":
                continue
            hr_str = row["HR"].strip()
            if not hr_str:
                continue
            d1 = row["Prior disorder"].strip().replace("Specific Phobia", "Specific phobia")
            d2 = row["Later disorder"].strip().replace("Specific Phobia", "Specific phobia")
            hr = float(hr_str)
            key = tuple(sorted([d1, d2]))
            pair_hrs[key].append(hr)

    G = nx.Graph()
    for (d1, d2), hrs in pair_hrs.items():
        max_hr = max(hrs)
        if max_hr <= 1.0:
            continue
        G.add_node(d1)
        G.add_node(d2)
        G.add_edge(d1, d2, weight=math.log(max_hr), hr=max_hr)
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
    data_path = Path(__file__).parent / ".." / ".." / ".." / ".." / "psychiatric-comorbidity-curvature" / "experiments" / "data" / "mcgrath2020" / "comorbidity_hazard_ratios.csv"
    if not data_path.exists():
        data_path = Path("../../data/mcgrath2020/comorbidity_hazard_ratios.csv")
    if not data_path.exists():
        data_path = Path(__file__).parent.parent / "data" / "mcgrath2020" / "comorbidity_hazard_ratios.csv"

    output_dir = Path("../../results/psych/data_driven")
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    print(f"[{ts}] McGrath 2020 comorbidity graph: ORC on WHO WMH hazard ratios")

    G = build_mcgrath_graph(str(data_path))
    n_possible = G.number_of_nodes() * (G.number_of_nodes() - 1) / 2
    density = G.number_of_edges() / n_possible if n_possible > 0 else 0
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, density={density:.2f}")

    print(f"\nTop-10 edges by HR:")
    for u, v, d in sorted(G.edges(data=True), key=lambda x: -x[2]["hr"])[:10]:
        et = classify_edge(u, v, CLUSTERS)
        print(f"  {u:20s}-{v:20s}  HR={d['hr']:6.1f}  log(HR)={d['weight']:.2f}  [{et}]")

    orc = ollivier_ricci_curvature(G, alpha=0.5)

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
        print(f"Mann-Whitney U={U}, p={p_mw:.6f}, Cohen's d={d:.3f}")
    else:
        U, p_mw, d = 0, 1.0, 0.0

    node_curv = {}
    for n in G.nodes():
        edges_n = [(n, nb) for nb in G.neighbors(n)]
        curv = [orc.get(e, orc.get((e[1], e[0]), 0)) for e in edges_n]
        node_curv[n] = round(float(np.mean(curv)), 4) if curv else 0

    print(f"\nNode ranking (lowest → highest curvature):")
    for n, k in sorted(node_curv.items(), key=lambda x: x[1])[:10]:
        cl = "?"
        for name, members in CLUSTERS.items():
            if n in members:
                cl = name
                break
        print(f"  {n:20s}  κ={k:+.4f}  deg={G.degree(n)}  [{cl}]")

    result = {
        "data_source": "McGrath et al. 2020, WHO World Mental Health Survey (N=145,990)",
        "graph_type": "comorbidity hazard ratios (log HR)",
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
    }

    out_path = output_dir / f"mcgrath_comorbidity_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
