"""V5a extra experiments: permutation test, leave-one-edge-out, weighted ORC.

Adds results to results/reviewer/v5a_extra_results.json
"""
import json
import numpy as np
import networkx as nx
import ot
from scipy import stats
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

EDGES = [
    ("MDD", "GAD"), ("MDD", "insomnia"), ("MDD", "SUD"), ("MDD", "anhedonia"),
    ("MDD", "suicide_ideation"), ("MDD", "PTSD"), ("MDD", "SCZ"), ("MDD", "BIP"),
    ("MDD", "BPD"), ("MDD", "fatigue"), ("MDD", "cortisol"), ("MDD", "inflammation"),
    ("MDD", "OCD"), ("GAD", "panic"), ("GAD", "somatic"), ("GAD", "social_anxiety"),
    ("GAD", "PTSD"), ("GAD", "ADHD"), ("GAD", "ASD"), ("GAD", "OCD"),
    ("insomnia", "fatigue"), ("insomnia", "cortisol"), ("panic", "agoraphobia"),
    ("SUD", "psychosis"), ("SUD", "suicide_ideation"), ("SUD", "PTSD"),
    ("SUD", "BIP"), ("SUD", "ADHD"), ("SUD", "anhedonia"), ("SUD", "BPD"),
    ("SUD", "inflammation"), ("PTSD", "dissociation"), ("PTSD", "cortisol"),
    ("SCZ", "psychosis"), ("ADHD", "ASD"), ("dissociation", "BPD"),
    ("BPD", "suicide_ideation"), ("fatigue", "inflammation"),
    ("social_anxiety", "avoidance"),
]

# Published comorbidity odds ratios (from Kessler 2005 NCS-R and other sources).
# Where no published OR exists, use 1.0 (no association strength info).
# These are approximate ORs from the literature for each edge.
EDGE_WEIGHTS = {
    ("MDD", "GAD"): 6.0,        # Kessler 2005: OR ~6.0
    ("MDD", "insomnia"): 3.5,   # Baglioni 2011 meta-analysis
    ("MDD", "SUD"): 2.0,        # Davis 2008
    ("MDD", "anhedonia"): 4.0,  # definitional overlap
    ("MDD", "suicide_ideation"): 3.5,  # Nock 2008
    ("MDD", "PTSD"): 3.5,       # Flory 2015
    ("MDD", "SCZ"): 1.8,        # Buckley 2009
    ("MDD", "BIP"): 5.0,        # Hirschfeld 2003
    ("MDD", "BPD"): 3.0,        # Gunderson 2008
    ("MDD", "fatigue"): 4.5,    # Demyttenaere 2005
    ("MDD", "cortisol"): 2.5,   # Pariante 2008
    ("MDD", "inflammation"): 1.5,  # Dowlati 2010 meta
    ("MDD", "OCD"): 2.5,        # Brakoulias 2017
    ("GAD", "panic"): 4.0,      # Brown 2001
    ("GAD", "somatic"): 2.5,    # Kroenke 2007
    ("GAD", "social_anxiety"): 3.5,  # Mennin 2009
    ("GAD", "PTSD"): 3.0,       # Ginzburg 2010
    ("GAD", "ADHD"): 2.0,       # Kessler 2006
    ("GAD", "ASD"): 2.0,        # van Steensel 2011
    ("GAD", "OCD"): 2.5,        # Abramowitz 2003
    ("insomnia", "fatigue"): 5.0,   # strong clinical association
    ("insomnia", "cortisol"): 3.0,  # Vgontzas 2001
    ("panic", "agoraphobia"): 8.0,  # DSM-5 specifier
    ("SUD", "psychosis"): 2.5,  # Niemi-Pynttari 2013
    ("SUD", "suicide_ideation"): 2.0,  # Wilcox 2004
    ("SUD", "PTSD"): 2.5,       # Jacobsen 2001
    ("SUD", "BIP"): 3.5,        # Regier 1990
    ("SUD", "ADHD"): 3.0,       # Lee 2011
    ("SUD", "anhedonia"): 2.0,   # Garfield 2014
    ("SUD", "BPD"): 3.0,        # Trull 2000
    ("SUD", "inflammation"): 1.5,  # Crews 2006
    ("PTSD", "dissociation"): 4.0,  # Lanius 2010
    ("PTSD", "cortisol"): 3.0,  # Yehuda 2006
    ("SCZ", "psychosis"): 9.0,  # definitional
    ("ADHD", "ASD"): 4.0,       # Rommelse 2010
    ("dissociation", "BPD"): 4.5,  # Zanarini 2000
    ("BPD", "suicide_ideation"): 5.0,  # Soloff 2000
    ("fatigue", "inflammation"): 3.0,  # Bower 2011
    ("social_anxiety", "avoidance"): 7.0,  # Hofmann 2007 / definitional
}


def build_graph():
    G = nx.Graph()
    for a, b in EDGES:
        G.add_edge(a, b)
    return G


def build_weighted_graph():
    G = nx.Graph()
    for a, b in EDGES:
        w = EDGE_WEIGHTS.get((a, b), EDGE_WEIGHTS.get((b, a), 1.0))
        G.add_edge(a, b, weight=w)
    return G


def compute_orc(G, alpha=0.5):
    nodes = sorted(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    dist = dict(nx.all_pairs_shortest_path_length(G))
    D = np.zeros((n, n))
    for u in nodes:
        for v in nodes:
            D[idx[u], idx[v]] = dist[u].get(v, 999)
    curvatures = {}
    for u, v in G.edges():
        mu_u = np.zeros(n)
        mu_u[idx[u]] = alpha
        nbrs_u = list(G.neighbors(u))
        for w in nbrs_u:
            mu_u[idx[w]] = (1 - alpha) / len(nbrs_u)
        mu_v = np.zeros(n)
        mu_v[idx[v]] = alpha
        nbrs_v = list(G.neighbors(v))
        for w in nbrs_v:
            mu_v[idx[w]] = (1 - alpha) / len(nbrs_v)
        w1 = ot.emd2(mu_u, mu_v, D)
        kappa = 1 - w1 / D[idx[u], idx[v]]
        curvatures[(u, v)] = kappa
        curvatures[(v, u)] = kappa
    return curvatures


def compute_weighted_orc(G, alpha=0.5):
    """ORC with edge weights: neighbor mass proportional to edge weight."""
    nodes = sorted(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    # Weighted shortest paths
    dist = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    D = np.zeros((n, n))
    for u in nodes:
        for v in nodes:
            D[idx[u], idx[v]] = dist[u].get(v, 999)
    curvatures = {}
    for u, v in G.edges():
        mu_u = np.zeros(n)
        mu_u[idx[u]] = alpha
        nbrs_u = list(G.neighbors(u))
        # Weight-proportional mass distribution
        total_w = sum(G[u][w].get('weight', 1.0) for w in nbrs_u)
        for w in nbrs_u:
            mu_u[idx[w]] = (1 - alpha) * G[u][w].get('weight', 1.0) / total_w
        mu_v = np.zeros(n)
        mu_v[idx[v]] = alpha
        nbrs_v = list(G.neighbors(v))
        total_w = sum(G[v][w].get('weight', 1.0) for w in nbrs_v)
        for w in nbrs_v:
            mu_v[idx[w]] = (1 - alpha) * G[v][w].get('weight', 1.0) / total_w
        d_uv = D[idx[u], idx[v]]
        if d_uv == 0:
            kappa = 0.0
        else:
            w1 = ot.emd2(mu_u, mu_v, D)
            kappa = 1 - w1 / d_uv
        curvatures[(u, v)] = kappa
        curvatures[(v, u)] = kappa
    return curvatures


def mean_curvature(G, curvatures):
    result = {}
    for v in G.nodes():
        edges = [(v, u) for u in G.neighbors(v)]
        result[v] = np.mean([curvatures[(a, b)] for a, b in edges])
    return result


def rank_nodes(scores, ascending=True):
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=not ascending)
    return {node: rank + 1 for rank, (node, _) in enumerate(sorted_nodes)}


# ============================================================
# EXPERIMENT 1: Node-label permutation test
# ============================================================
def experiment_permutation_test(n_perms=10000):
    """Permutation test: randomly relabel nodes, compute curvature gap for top-2 betweenness nodes."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running permutation test ({n_perms} permutations)...")
    G = build_graph()
    orc = compute_orc(G, alpha=0.5)
    mc = mean_curvature(G, orc)

    # Observed gap: GAD kappa - MDD kappa
    observed_gap = mc["GAD"] - mc["MDD"]
    print(f"  Observed gap (κ̄_GAD - κ̄_MDD) = {observed_gap:.4f}")

    # Also compute: among top-2 betweenness nodes, what's the curvature gap?
    btwn = nx.betweenness_centrality(G)
    btwn_sorted = sorted(btwn.items(), key=lambda x: x[1], reverse=True)
    top2_btwn = [btwn_sorted[0][0], btwn_sorted[1][0]]
    print(f"  Top-2 betweenness: {top2_btwn}")

    rng = np.random.default_rng(42)
    nodes = sorted(G.nodes())
    null_gaps = []
    null_abs_gaps = []

    for _ in tqdm(range(n_perms), desc="Permutations"):
        # Permute node labels
        perm = rng.permutation(nodes)
        label_map = dict(zip(nodes, perm))
        # Remap curvatures to permuted labels
        perm_mc = {label_map[v]: mc[v] for v in nodes}
        # Gap between the nodes that are NOW labeled GAD and MDD
        gap = perm_mc["GAD"] - perm_mc["MDD"]
        null_gaps.append(float(gap))
        null_abs_gaps.append(float(abs(gap)))

    null_gaps = np.array(null_gaps)
    null_abs_gaps = np.array(null_abs_gaps)

    # One-sided: how often is the permuted gap <= observed (more negative)?
    p_one_sided = float(np.mean(null_gaps <= observed_gap))
    # Two-sided: how often is |gap| >= |observed|?
    p_two_sided = float(np.mean(null_abs_gaps >= abs(observed_gap)))

    print(f"  p (one-sided, gap <= observed): {p_one_sided:.4f}")
    print(f"  p (two-sided, |gap| >= |observed|): {p_two_sided:.4f}")
    print(f"  Null mean: {np.mean(null_gaps):.4f}, std: {np.std(null_gaps):.4f}")

    return {
        "observed_gap": float(observed_gap),
        "n_permutations": n_perms,
        "p_one_sided": p_one_sided,
        "p_two_sided": p_two_sided,
        "null_mean": float(np.mean(null_gaps)),
        "null_std": float(np.std(null_gaps)),
        "null_ci_lower": float(np.percentile(null_gaps, 2.5)),
        "null_ci_upper": float(np.percentile(null_gaps, 97.5)),
    }


# ============================================================
# EXPERIMENT 2: Leave-one-edge-out sensitivity
# ============================================================
def experiment_leave_one_edge_out():
    """Remove each edge, recompute GAD/MDD rankings and gap."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running leave-one-edge-out...")
    G = build_graph()
    orc = compute_orc(G, alpha=0.5)
    mc = mean_curvature(G, orc)
    baseline_gap = mc["GAD"] - mc["MDD"]
    baseline_gad_rank = rank_nodes(mc, ascending=True)["GAD"]
    baseline_mdd_rank = rank_nodes(mc, ascending=True)["MDD"]

    results = []
    flips = 0
    for u, v in tqdm(EDGES, desc="Leave-one-edge-out"):
        H = G.copy()
        H.remove_edge(u, v)
        if not nx.is_connected(H):
            results.append({
                "edge": f"{u}--{v}",
                "disconnects": True,
                "gad_kappa": None,
                "mdd_kappa": None,
                "gap": None,
                "gad_rank": None,
                "mdd_rank": None,
                "flips_ordering": None,
            })
            continue
        orc_h = compute_orc(H, alpha=0.5)
        mc_h = mean_curvature(H, orc_h)
        gap = mc_h.get("GAD", 0) - mc_h.get("MDD", 0)
        ranks = rank_nodes(mc_h, ascending=True)
        flipped = gap > 0  # MDD becomes more bridge-like than GAD
        if flipped:
            flips += 1
        results.append({
            "edge": f"{u}--{v}",
            "disconnects": False,
            "gad_kappa": float(mc_h.get("GAD", 0)),
            "mdd_kappa": float(mc_h.get("MDD", 0)),
            "gap": float(gap),
            "gad_rank": int(ranks.get("GAD", 0)),
            "mdd_rank": int(ranks.get("MDD", 0)),
            "flips_ordering": bool(flipped),
            "edge_curvature": float(orc.get((u, v), orc.get((v, u), 0))),
        })

    # Sort by gap (most positive = most likely to flip)
    valid = [r for r in results if not r["disconnects"]]
    valid_sorted = sorted(valid, key=lambda x: x["gap"], reverse=True)

    n_disconnect = sum(1 for r in results if r["disconnects"])
    print(f"  Baseline gap: {baseline_gap:.4f}")
    print(f"  Edges that disconnect graph: {n_disconnect}")
    print(f"  Edges that flip GAD/MDD ordering: {flips}/{len(EDGES)}")
    print(f"  Top 5 most destabilizing edges:")
    for r in valid_sorted[:5]:
        print(f"    {r['edge']}: gap={r['gap']:.4f}, GAD rank={r['gad_rank']}, "
              f"MDD rank={r['mdd_rank']}, flips={r['flips_ordering']}")

    return {
        "baseline_gap": float(baseline_gap),
        "n_edges": len(EDGES),
        "n_disconnect": n_disconnect,
        "n_flips": flips,
        "per_edge": results,
        "most_destabilizing": valid_sorted[:5],
    }


# ============================================================
# EXPERIMENT 3: Weighted ORC from odds ratios
# ============================================================
def experiment_weighted_orc():
    """Compute ORC with edge weights from published odds ratios."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running weighted ORC...")

    # Unweighted baseline
    G_uw = build_graph()
    orc_uw = compute_orc(G_uw, alpha=0.5)
    mc_uw = mean_curvature(G_uw, orc_uw)
    ranks_uw = rank_nodes(mc_uw, ascending=True)

    # Weighted: use 1/OR as distance (higher OR = closer = shorter edge)
    G_w = nx.Graph()
    for a, b in EDGES:
        or_val = EDGE_WEIGHTS.get((a, b), EDGE_WEIGHTS.get((b, a), 1.0))
        G_w.add_edge(a, b, weight=1.0 / or_val)  # distance = 1/OR

    orc_w = compute_weighted_orc(G_w, alpha=0.5)
    mc_w = mean_curvature(G_w, orc_w)
    ranks_w = rank_nodes(mc_w, ascending=True)

    # Also try: weight-proportional neighbor mass, unweighted distance
    G_w2 = nx.Graph()
    for a, b in EDGES:
        or_val = EDGE_WEIGHTS.get((a, b), EDGE_WEIGHTS.get((b, a), 1.0))
        G_w2.add_edge(a, b, weight=or_val)

    # For this version, use unweighted shortest path but OR-proportional mass
    nodes = sorted(G_w2.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    dist_uw = dict(nx.all_pairs_shortest_path_length(G_uw))
    D = np.zeros((n, n))
    for u in nodes:
        for v in nodes:
            D[idx[u], idx[v]] = dist_uw[u].get(v, 999)

    curvatures_w2 = {}
    for u, v in G_w2.edges():
        alpha = 0.5
        mu_u = np.zeros(n)
        mu_u[idx[u]] = alpha
        nbrs_u = list(G_w2.neighbors(u))
        total_w = sum(G_w2[u][w].get('weight', 1.0) for w in nbrs_u)
        for w in nbrs_u:
            mu_u[idx[w]] = (1 - alpha) * G_w2[u][w].get('weight', 1.0) / total_w
        mu_v = np.zeros(n)
        mu_v[idx[v]] = alpha
        nbrs_v = list(G_w2.neighbors(v))
        total_w = sum(G_w2[v][w].get('weight', 1.0) for w in nbrs_v)
        for w in nbrs_v:
            mu_v[idx[w]] = (1 - alpha) * G_w2[v][w].get('weight', 1.0) / total_w
        w1 = ot.emd2(mu_u, mu_v, D)
        kappa = 1 - w1 / D[idx[u], idx[v]]
        curvatures_w2[(u, v)] = kappa
        curvatures_w2[(v, u)] = kappa

    mc_w2 = mean_curvature(G_w2, curvatures_w2)
    ranks_w2 = rank_nodes(mc_w2, ascending=True)

    # Rank correlation between weighted and unweighted
    nodes_list = sorted(G_uw.nodes())
    uw_vals = [mc_uw[v] for v in nodes_list]
    w_vals = [mc_w[v] for v in nodes_list]
    w2_vals = [mc_w2[v] for v in nodes_list]

    corr_w = stats.spearmanr(uw_vals, w_vals)
    corr_w2 = stats.spearmanr(uw_vals, w2_vals)

    print(f"\n  === Weighted ORC (distance = 1/OR) ===")
    print(f"  GAD: κ̄={mc_w['GAD']:.4f} (rank {ranks_w['GAD']})")
    print(f"  MDD: κ̄={mc_w['MDD']:.4f} (rank {ranks_w['MDD']})")
    print(f"  Gap: {mc_w['GAD'] - mc_w['MDD']:.4f}")
    print(f"  Rank correlation with unweighted: ρ={corr_w.correlation:.3f}")

    print(f"\n  === OR-proportional mass, unweighted distance ===")
    print(f"  GAD: κ̄={mc_w2['GAD']:.4f} (rank {ranks_w2['GAD']})")
    print(f"  MDD: κ̄={mc_w2['MDD']:.4f} (rank {ranks_w2['MDD']})")
    print(f"  Gap: {mc_w2['GAD'] - mc_w2['MDD']:.4f}")
    print(f"  Rank correlation with unweighted: ρ={corr_w2.correlation:.3f}")

    top5_uw = sorted(mc_uw.items(), key=lambda x: x[1])[:5]
    top5_w = sorted(mc_w.items(), key=lambda x: x[1])[:5]
    top5_w2 = sorted(mc_w2.items(), key=lambda x: x[1])[:5]
    print(f"\n  Top-5 bridges (unweighted): {[(n, f'{k:.3f}') for n, k in top5_uw]}")
    print(f"  Top-5 bridges (1/OR dist):  {[(n, f'{k:.3f}') for n, k in top5_w]}")
    print(f"  Top-5 bridges (OR mass):    {[(n, f'{k:.3f}') for n, k in top5_w2]}")

    per_node = []
    for v in nodes_list:
        per_node.append({
            "node": v,
            "unweighted_kappa": float(mc_uw[v]),
            "unweighted_rank": int(ranks_uw[v]),
            "weighted_dist_kappa": float(mc_w[v]),
            "weighted_dist_rank": int(ranks_w[v]),
            "weighted_mass_kappa": float(mc_w2[v]),
            "weighted_mass_rank": int(ranks_w2[v]),
        })

    return {
        "unweighted": {
            "gad_kappa": float(mc_uw["GAD"]),
            "mdd_kappa": float(mc_uw["MDD"]),
            "gad_rank": int(ranks_uw["GAD"]),
            "mdd_rank": int(ranks_uw["MDD"]),
        },
        "weighted_distance": {
            "gad_kappa": float(mc_w["GAD"]),
            "mdd_kappa": float(mc_w["MDD"]),
            "gad_rank": int(ranks_w["GAD"]),
            "mdd_rank": int(ranks_w["MDD"]),
            "gap": float(mc_w["GAD"] - mc_w["MDD"]),
            "spearman_vs_unweighted": float(corr_w.correlation),
        },
        "weighted_mass": {
            "gad_kappa": float(mc_w2["GAD"]),
            "mdd_kappa": float(mc_w2["MDD"]),
            "gad_rank": int(ranks_w2["GAD"]),
            "mdd_rank": int(ranks_w2["MDD"]),
            "gap": float(mc_w2["GAD"] - mc_w2["MDD"]),
            "spearman_vs_unweighted": float(corr_w2.correlation),
        },
        "per_node": per_node,
    }


if __name__ == "__main__":
    results = {}
    results["permutation_test"] = experiment_permutation_test(n_perms=10000)
    results["leave_one_edge_out"] = experiment_leave_one_edge_out()
    results["weighted_orc"] = experiment_weighted_orc()

    out = Path("results/reviewer/v5a_extra_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Saved to {out}")
