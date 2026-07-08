"""Cross-domain replication: run weighted ORC on psychiatric, autoimmune,
and cardiometabolic LDSC genetic correlation graphs.

Tests whether the curvature pattern (within-domain > cross-domain) replicates
across three independent disease domains using published LDSC rg values.
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


DOMAIN_CONFIGS = {
    "psychiatric": {
        "file": "data/ldsc_psychiatric_rg.tsv",
        "clusters": {
            "internalizing": {"MDD", "ANX", "PTSD"},
            "psychotic": {"SCZ", "BD"},
            "neurodevelopmental": {"ADHD", "ASD", "TS"},
            "compulsive": {"AN", "OCD"},
            "substance": {"ALCH"},
        },
        "sources": "Lee2019, Grotzinger2022",
    },
    "autoimmune": {
        "file": "data/autoimmune_ldsc_rg.tsv",
        "clusters": {
            "IBD": {"Crohn", "UC"},
            "connective_tissue": {"RA", "Lupus", "AS"},
            "barrier": {"Psoriasis", "Celiac"},
            "autoimmune_endo": {"T1D", "MS"},
        },
        "sources": "BulikSullivan2015, Ellinghaus2016",
    },
    "cardiometabolic": {
        "file": "data/cardiometabolic_ldsc_rg.tsv",
        "clusters": {
            "disease": {"CAD", "T2D"},
            "adiposity": {"BMI", "WHR"},
            "lipids": {"LDL", "HDL", "TG"},
            "blood_pressure": {"SBP", "DBP"},
        },
        "sources": "BulikSullivan2015, Evangelou2018",
    },
}


def build_weighted_graph(rg_file, p_threshold=0.05):
    G = nx.Graph()
    with open(rg_file) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            t1 = row["trait1"].strip()
            t2 = row["trait2"].strip()
            rg = float(row["rg"])
            se = float(row["se"])
            p = float(row["p"])
            if p > p_threshold:
                continue
            G.add_node(t1)
            G.add_node(t2)
            G.add_edge(t1, t2, weight=abs(rg), rg=rg, se=se, p=p)
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


def analyze_domain(domain_name, config, output_dir):
    print(f"\n{'='*60}")
    print(f"  {domain_name.upper()}")
    print(f"{'='*60}")

    G = build_weighted_graph(config["file"])
    n_possible = G.number_of_nodes() * (G.number_of_nodes() - 1) / 2
    density = G.number_of_edges() / n_possible if n_possible > 0 else 0
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, density={density:.2f}")

    orc = ollivier_ricci_curvature(G, alpha=0.5)

    clusters = config["clusters"]
    cross_curv = []
    within_curv = []
    for (u, v), k in orc.items():
        et = classify_edge(u, v, clusters)
        if et == "within":
            within_curv.append(k)
        else:
            cross_curv.append(k)

    result = {
        "domain": domain_name,
        "sources": config["sources"],
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": round(density, 3),
        "n_within": len(within_curv),
        "n_cross": len(cross_curv),
    }

    if within_curv and cross_curv:
        U, p_mw = stats.mannwhitneyu(cross_curv, within_curv, alternative="two-sided")
        d = (np.mean(within_curv) - np.mean(cross_curv)) / np.sqrt(
            (np.std(within_curv)**2 + np.std(cross_curv)**2) / 2
        ) if (np.std(within_curv) > 0 or np.std(cross_curv) > 0) else 0

        result.update({
            "within_mean": round(float(np.mean(within_curv)), 4),
            "cross_mean": round(float(np.mean(cross_curv)), 4),
            "within_std": round(float(np.std(within_curv)), 4),
            "cross_std": round(float(np.std(cross_curv)), 4),
            "U": float(U),
            "p": round(float(p_mw), 6),
            "cohens_d": round(float(d), 3),
        })
        print(f"  Within-cluster: κ={np.mean(within_curv):.4f} ± {np.std(within_curv):.4f} (n={len(within_curv)})")
        print(f"  Cross-cluster:  κ={np.mean(cross_curv):.4f} ± {np.std(cross_curv):.4f} (n={len(cross_curv)})")
        print(f"  Mann-Whitney p={p_mw:.4f}, Cohen's d={d:.3f}")
    else:
        print(f"  WARNING: within={len(within_curv)}, cross={len(cross_curv)}")

    # Node curvature ranking
    node_curv = {}
    for n in G.nodes():
        edges = [(n, nb) for nb in G.neighbors(n)]
        curv = [orc.get(e, orc.get((e[1], e[0]), 0)) for e in edges]
        node_curv[n] = round(float(np.mean(curv)), 4) if curv else 0

    result["node_curvature"] = dict(sorted(node_curv.items(), key=lambda x: x[1]))

    print(f"\n  Node ranking (lowest → highest curvature):")
    for n, k in sorted(node_curv.items(), key=lambda x: x[1])[:5]:
        print(f"    {n:15s}  κ={k:+.4f}  deg={G.degree(n)}")

    # Edge details
    result["edges"] = {}
    for (u, v), k in sorted(orc.items(), key=lambda x: x[1]):
        et = classify_edge(u, v, clusters)
        result["edges"][f"{u}-{v}"] = {
            "kappa": round(k, 4),
            "rg": G[u][v].get("rg", 0),
            "type": et,
        }

    return result


def make_comparison_figure(results, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, (domain, r) in zip(axes, results.items()):
        within = r.get("within_mean", 0)
        cross = r.get("cross_mean", 0)
        within_std = r.get("within_std", 0)
        cross_std = r.get("cross_std", 0)
        n_w = r.get("n_within", 0)
        n_c = r.get("n_cross", 0)
        p = r.get("p", 1.0)

        bars = ax.bar(["Within\ncluster", "Cross\ncluster"], [within, cross],
                      yerr=[within_std / max(n_w**0.5, 1), cross_std / max(n_c**0.5, 1)],
                      color=["#2196F3", "#FF9800"], capsize=5, width=0.5)

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        y_max = max(within + within_std, cross + cross_std)
        ax.text(0.5, y_max * 1.15, f"p={p:.3f} {sig}", ha="center", fontsize=10)

        ax.set_title(f"{domain.capitalize()}\n({r['n_nodes']} traits, {r['n_edges']} edges)", fontsize=11)
        ax.set_ylabel("Mean ORC (κ)")
        ax.set_ylim(bottom=min(0, cross - cross_std * 1.5))

    plt.tight_layout()
    out_path = output_dir / "fig_cross_domain_comparison.pdf"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\nFigure saved to {out_path}")
    return str(out_path)


def main():
    output_dir = Path("../../results/psych/data_driven")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    print(f"[{ts}] Cross-domain LDSC curvature replication")

    all_results = {}
    for domain_name, config in DOMAIN_CONFIGS.items():
        all_results[domain_name] = analyze_domain(domain_name, config, output_dir)

    # Summary comparison
    print(f"\n{'='*60}")
    print("  CROSS-DOMAIN SUMMARY")
    print(f"{'='*60}")
    print(f"{'Domain':20s} {'Within κ':>10s} {'Cross κ':>10s} {'d':>8s} {'p':>10s}")
    print("-" * 60)
    for domain, r in all_results.items():
        w = r.get("within_mean", float("nan"))
        c = r.get("cross_mean", float("nan"))
        d = r.get("cohens_d", float("nan"))
        p = r.get("p", float("nan"))
        print(f"{domain:20s} {w:>10.4f} {c:>10.4f} {d:>8.3f} {p:>10.4f}")

    # Replication test: is the within > cross pattern consistent?
    n_replicated = sum(1 for r in all_results.values()
                       if r.get("within_mean", 0) > r.get("cross_mean", 0))
    n_significant = sum(1 for r in all_results.values()
                        if r.get("p", 1) < 0.05)
    print(f"\nPattern replication: {n_replicated}/3 domains show within > cross")
    print(f"Statistical significance: {n_significant}/3 domains reach p < 0.05")

    all_results["summary"] = {
        "timestamp": ts,
        "n_domains": len(all_results) - 1,
        "n_replicated": n_replicated,
        "n_significant": n_significant,
    }

    out_path = output_dir / f"cross_domain_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    make_comparison_figure(
        {k: v for k, v in all_results.items() if k != "summary"},
        output_dir / "figures"
    )


if __name__ == "__main__":
    main()
