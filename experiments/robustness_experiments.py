"""Robustness experiments for the curvature paper.

Three experiments:
  1. Catalog perturbation: subsample 40/47 entries 500 times, recompute
     edge-level curvature z-scores, report fraction of iterations each
     key edge survives at |z| > 1.96.
  2. Leave-one-family-out: drop each of 9 disorder families, recompute
     curvature + permutation null, check if bottleneck edges survive.
  3. Edge betweenness comparison: compute edge betweenness centrality,
     correlate with edge curvature, test whether ORC identifies edges
     that betweenness misses.

Usage:
    uv run --no-project --with numpy --with scipy --with networkx --with pot --with tqdm --with matplotlib \
        python psych/experiments/robustness_experiments.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import networkx as nx
import ot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from catalog_data import ENTRIES, SHARED_NODES, FAMILIES, VERDICTS


def build_graph(entries=None):
    if entries is None:
        entries = ENTRIES
    G = nx.Graph()
    for fam in FAMILIES:
        G.add_node(f"family:{fam}", layer="family")

    used_mechanisms = set()
    for entry in entries:
        for node in entry["shared_nodes"]:
            used_mechanisms.add(node)

    for node_name, info in SHARED_NODES.items():
        if node_name in used_mechanisms:
            G.add_node(f"mechanism:{node_name}", layer="mechanism",
                       n_disorders=len(info["disorders"]))
            for disorder in info["disorders"]:
                G.add_edge(f"family:{disorder}", f"mechanism:{node_name}")

    for entry in entries:
        eid = f"claim:{entry['id']}"
        G.add_node(eid, layer="claim",
                   verdict=entry["verdict"],
                   verdict_score=VERDICTS.get(entry["verdict"], 2),
                   family=entry["family"])
        G.add_edge(eid, f"family:{entry['family']}")
        for node in entry["shared_nodes"]:
            if f"mechanism:{node}" in G:
                G.add_edge(eid, f"mechanism:{node}")

    G.remove_nodes_from(list(nx.isolates(G)))
    return G


def compute_curvatures(G, alpha=0.5):
    sp = dict(nx.all_pairs_shortest_path_length(G))
    curvatures = {}
    for u, v in G.edges():
        nb_u = list(G.neighbors(u))
        nb_v = list(G.neighbors(v))
        all_nodes = sorted(set([u] + nb_u + [v] + nb_v))
        idx = {n: i for i, n in enumerate(all_nodes)}

        mu = np.zeros(len(all_nodes))
        mu[idx[u]] = alpha
        for nb in nb_u:
            mu[idx[nb]] += (1 - alpha) / len(nb_u)

        nu = np.zeros(len(all_nodes))
        nu[idx[v]] = alpha
        for nb in nb_v:
            nu[idx[nb]] += (1 - alpha) / len(nb_v)

        cost = np.zeros((len(all_nodes), len(all_nodes)))
        for i, ni in enumerate(all_nodes):
            for j, nj in enumerate(all_nodes):
                cost[i, j] = sp.get(ni, {}).get(nj, 100)

        W1 = ot.emd2(mu, nu, cost)
        d_uv = sp[u][v]
        curvatures[(u, v)] = float(1.0 - W1 / d_uv) if d_uv > 0 else 0.0
    return curvatures


def degree_preserving_rewire(G, n_swaps=None):
    H = G.copy()
    if n_swaps is None:
        n_swaps = H.number_of_edges() * 10
    nx.double_edge_swap(H, nswap=n_swaps, max_tries=n_swaps * 10, seed=None)
    return H


def edge_zscores(G, n_perms=100):
    """Compute per-edge z-scores against degree-preserving null."""
    obs_curvatures = compute_curvatures(G)
    edges = list(G.edges())
    null_curvatures = {e: [] for e in edges}

    for _ in range(n_perms):
        try:
            H = degree_preserving_rewire(G)
            perm_curv = compute_curvatures(H)
            for u, v in edges:
                c = perm_curv.get((u, v), perm_curv.get((v, u), None))
                if c is not None:
                    null_curvatures[(u, v)].append(c)
        except nx.NetworkXError:
            continue

    results = {}
    for (u, v) in edges:
        obs = obs_curvatures.get((u, v), obs_curvatures.get((v, u), 0))
        null = np.array(null_curvatures[(u, v)])
        if len(null) < 10:
            continue
        null_mean = np.mean(null)
        null_std = np.std(null)
        z = (obs - null_mean) / null_std if null_std > 0 else 0.0
        results[(u, v)] = {"obs": obs, "null_mean": null_mean, "null_std": null_std, "z": z}
    return results


def edge_key_short(u, v):
    """Human-readable edge label."""
    def short(n):
        if n.startswith("family:"):
            return n.replace("family:", "")
        if n.startswith("mechanism:"):
            return n.replace("mechanism:", "")
        if n.startswith("claim:"):
            return n.replace("claim:", "c")
        return n
    return f"{short(u)}--{short(v)}"


KEY_EDGES = [
    ("family:Depression", "mechanism:HPA_axis"),
    ("family:Depression", "mechanism:circadian"),
    ("family:Depression", "mechanism:serotonin"),
    ("family:Addiction", "mechanism:HPA_axis"),
    ("family:Addiction", "mechanism:dopamine"),
    ("mechanism:circadian", "claim:031b"),
    ("claim:004", "mechanism:FKBP5"),
    ("mechanism:MIA_prenatal", "claim:029"),
    ("mechanism:MIA_prenatal", "claim:028"),
    ("mechanism:synaptic_pruning", "claim:039"),
    ("mechanism:CSTC_circuit", "claim:034"),
]


def experiment_1_catalog_perturbation(n_bootstrap=500, subsample_size=40, n_null_perms=50):
    """Subsample 40/47 entries, rebuild graph, recompute edge z-scores."""
    print(f"\n{'='*80}")
    print(f"  EXPERIMENT 1: Catalog Perturbation ({n_bootstrap} iterations, {subsample_size}/{len(ENTRIES)} entries)")
    print(f"{'='*80}")

    edge_survival = {}
    edge_z_distributions = {}

    for i in tqdm(range(n_bootstrap), desc="Bootstrap iterations"):
        indices = np.random.choice(len(ENTRIES), size=subsample_size, replace=False)
        subset = [ENTRIES[j] for j in indices]
        G_sub = build_graph(subset)

        if G_sub.number_of_edges() < 5:
            continue

        obs_curvatures = compute_curvatures(G_sub)

        null_curvatures = {e: [] for e in G_sub.edges()}
        for _ in range(n_null_perms):
            try:
                H = degree_preserving_rewire(G_sub)
                pc = compute_curvatures(H)
                for u, v in G_sub.edges():
                    c = pc.get((u, v), pc.get((v, u), None))
                    if c is not None:
                        null_curvatures[(u, v)].append(c)
            except nx.NetworkXError:
                continue

        for u, v in G_sub.edges():
            obs = obs_curvatures.get((u, v), obs_curvatures.get((v, u), 0))
            null = np.array(null_curvatures[(u, v)])
            if len(null) < 5:
                continue
            z = (obs - np.mean(null)) / np.std(null) if np.std(null) > 0 else 0.0

            ek = edge_key_short(u, v)
            ek_rev = edge_key_short(v, u)
            key = ek if ek < ek_rev else ek_rev
            if key not in edge_survival:
                edge_survival[key] = {"present": 0, "significant": 0, "z_values": []}
            edge_survival[key]["present"] += 1
            edge_survival[key]["z_values"].append(z)
            if abs(z) > 1.96:
                edge_survival[key]["significant"] += 1

    print(f"\n  Key edge survival rates (present in subsample & |z|>1.96):")
    key_edge_labels = set()
    for u, v in KEY_EDGES:
        ek = edge_key_short(u, v)
        ek_rev = edge_key_short(v, u)
        key_edge_labels.add(ek if ek < ek_rev else ek_rev)

    results = {}
    for key in sorted(edge_survival.keys()):
        s = edge_survival[key]
        if s["present"] < 10:
            continue
        rate = s["significant"] / s["present"]
        mean_z = np.mean(s["z_values"])
        marker = " <-- KEY" if key in key_edge_labels else ""
        if key in key_edge_labels or rate > 0.5:
            print(f"    {key:50s}: {s['significant']:3d}/{s['present']:3d} = {rate:.1%}  mean_z={mean_z:+.2f}{marker}")
        results[key] = {
            "n_present": s["present"],
            "n_significant": s["significant"],
            "survival_rate": rate,
            "mean_z": float(mean_z),
            "std_z": float(np.std(s["z_values"])),
        }
    return results


def experiment_2_leave_one_family_out(n_null_perms=100):
    """Drop each disorder family, recompute curvature z-scores."""
    print(f"\n{'='*80}")
    print(f"  EXPERIMENT 2: Leave-One-Family-Out")
    print(f"{'='*80}")

    full_G = build_graph()
    full_zscores = edge_zscores(full_G, n_perms=n_null_perms)

    results = {}
    for drop_family in FAMILIES:
        subset = [e for e in ENTRIES if e["family"] != drop_family]
        G_sub = build_graph(subset)
        print(f"\n  Dropping {drop_family}: {G_sub.number_of_nodes()} nodes, {G_sub.number_of_edges()} edges")

        sub_zscores = edge_zscores(G_sub, n_perms=n_null_perms)

        surviving = []
        lost = []
        for (u, v), data in sub_zscores.items():
            ek = edge_key_short(u, v)
            if abs(data["z"]) > 1.96:
                surviving.append((ek, data["z"]))

        for (u, v), data in full_zscores.items():
            if abs(data["z"]) > 1.96:
                ek = edge_key_short(u, v)
                found = any(edge_key_short(su, sv) == ek or edge_key_short(sv, su) == ek
                           for (su, sv) in sub_zscores)
                if not found:
                    lost.append((ek, data["z"]))

        print(f"    Surviving significant edges: {len(surviving)}")
        for ek, z in sorted(surviving, key=lambda x: x[1]):
            print(f"      {ek:50s}: z={z:+.2f}")
        if lost:
            print(f"    Lost edges (only exist with {drop_family}):")
            for ek, z in lost:
                print(f"      {ek:50s}: z={z:+.2f}")

        results[drop_family] = {
            "n_nodes": G_sub.number_of_nodes(),
            "n_edges": G_sub.number_of_edges(),
            "n_surviving_sig": len(surviving),
            "n_lost": len(lost),
            "surviving": {ek: float(z) for ek, z in surviving},
            "lost": {ek: float(z) for ek, z in lost},
        }
    return results


def experiment_3_betweenness_comparison():
    """Compare edge curvature with edge betweenness centrality."""
    print(f"\n{'='*80}")
    print(f"  EXPERIMENT 3: Edge Betweenness vs Edge Curvature")
    print(f"{'='*80}")

    G = build_graph()
    obs_curvatures = compute_curvatures(G)
    edge_betweenness = nx.edge_betweenness_centrality(G)

    curvs = []
    betws = []
    labels = []
    for (u, v) in G.edges():
        c = obs_curvatures.get((u, v), obs_curvatures.get((v, u), None))
        b = edge_betweenness.get((u, v), edge_betweenness.get((v, u), None))
        if c is not None and b is not None:
            curvs.append(c)
            betws.append(b)
            labels.append(edge_key_short(u, v))

    curvs = np.array(curvs)
    betws = np.array(betws)

    r, p = stats.pearsonr(curvs, betws)
    rho, p_rho = stats.spearmanr(curvs, betws)
    print(f"\n  Pearson  r = {r:.3f}, p = {p:.4f}")
    print(f"  Spearman ρ = {rho:.3f}, p = {p_rho:.4f}")

    # Find edges where curvature and betweenness disagree
    curv_rank = stats.rankdata(-curvs)  # high curvature = low rank
    betw_rank = stats.rankdata(-betws)  # high betweenness = low rank
    rank_diff = curv_rank - betw_rank

    print(f"\n  Edges where curvature identifies anomaly but betweenness does not:")
    print(f"  (high negative curvature but low betweenness)")
    disagree = sorted(zip(rank_diff, labels, curvs, betws), key=lambda x: x[0])
    for rd, label, c, b in disagree[:10]:
        print(f"    {label:50s}: curv={c:+.4f} betw={b:.4f} rank_diff={rd:+.0f}")

    print(f"\n  Edges where betweenness ranks high but curvature does not:")
    for rd, label, c, b in disagree[-10:]:
        print(f"    {label:50s}: curv={c:+.4f} betw={b:.4f} rank_diff={rd:+.0f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(betws, curvs, alpha=0.5, s=20, c="steelblue")
    ax.set_xlabel("Edge betweenness centrality", fontsize=12)
    ax.set_ylabel("Ollivier-Ricci curvature", fontsize=12)
    ax.set_title(f"Edge betweenness vs curvature (r = {r:.3f}, ρ = {rho:.3f})", fontsize=13)

    # Label outlier edges
    for i, label in enumerate(labels):
        if abs(rank_diff[i]) > len(labels) * 0.6:
            ax.annotate(label, (betws[i], curvs[i]), fontsize=6, alpha=0.7,
                       xytext=(5, 5), textcoords="offset points")

    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig_path = Path("psych/experiments/figures/fig_betweenness_vs_curvature.pdf")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {fig_path}")

    return {
        "pearson_r": float(r), "pearson_p": float(p),
        "spearman_rho": float(rho), "spearman_p": float(p_rho),
        "n_edges": len(curvs),
    }


def main():
    print(f"[{datetime.now():%H:%M:%S}] Starting robustness experiments")
    print(f"  Catalog: {len(ENTRIES)} entries, {len(FAMILIES)} families")

    results = {}

    # Experiment 3 is fast — run first
    results["betweenness_comparison"] = experiment_3_betweenness_comparison()

    # Experiment 1: catalog perturbation (this is the heavy one)
    results["catalog_perturbation"] = experiment_1_catalog_perturbation(
        n_bootstrap=500, subsample_size=40, n_null_perms=50
    )

    # Experiment 2: leave-one-family-out
    results["leave_one_family_out"] = experiment_2_leave_one_family_out(n_null_perms=100)

    # Save all results
    out_dir = Path("results/psych/psych/robustness")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"robustness_experiments_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[{datetime.now():%H:%M:%S}] All results saved: {out_path}")


if __name__ == "__main__":
    main()
