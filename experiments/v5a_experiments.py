"""V5a reviewer experiments: paired bootstrap test, ORC-degree correlation, modularity scores.

Adds results to results/reviewer/v5a_results.json
"""
import json
import numpy as np
import networkx as nx
import ot
import community as community_louvain
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


def build_graph():
    G = nx.Graph()
    for a, b in EDGES:
        G.add_edge(a, b)
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


def mean_curvature(G, curvatures):
    result = {}
    for v in G.nodes():
        edges = [(v, u) for u in G.neighbors(v)]
        result[v] = np.mean([curvatures[(a, b)] for a, b in edges])
    return result


def perturb_graph(G, rng):
    edges = list(G.edges())
    n_remove = rng.integers(1, max(2, int(0.2 * len(edges)) + 1))
    non_edges = list(nx.non_edges(G))
    n_add = rng.integers(0, max(1, int(0.1 * len(non_edges)) + 1))
    remove_idx = rng.choice(len(edges), size=min(n_remove, len(edges)), replace=False)
    H = G.copy()
    for i in remove_idx:
        u, v = edges[i]
        H.remove_edge(u, v)
        if not nx.is_connected(H) or min(dict(H.degree()).values()) < 1:
            H.add_edge(u, v)
    if non_edges and n_add > 0:
        add_idx = rng.choice(len(non_edges), size=min(n_add, len(non_edges)), replace=False)
        for i in add_idx:
            u, v = non_edges[i]
            H.add_edge(u, v)
    return H


def experiment_paired_bootstrap(n_trials=1000):
    """Paired bootstrap test on (κ̄_GAD − κ̄_MDD) across perturbation trials."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running paired bootstrap ({n_trials} trials)...")
    G = build_graph()
    rng = np.random.default_rng(42)

    gad_kappas = []
    mdd_kappas = []
    diffs = []

    for _ in tqdm(range(n_trials), desc="Bootstrap trials"):
        H = perturb_graph(G, rng)
        orc = compute_orc(H, alpha=0.5)
        mc = mean_curvature(H, orc)
        gad_k = mc.get("GAD", 0.0)
        mdd_k = mc.get("MDD", 0.0)
        gad_kappas.append(float(gad_k))
        mdd_kappas.append(float(mdd_k))
        diffs.append(float(gad_k - mdd_k))

    diffs = np.array(diffs)
    gad_kappas = np.array(gad_kappas)
    mdd_kappas = np.array(mdd_kappas)

    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))
    mean_diff = float(np.mean(diffs))
    frac_negative = float(np.mean(diffs < 0))

    print(f"  Mean(κ̄_GAD − κ̄_MDD) = {mean_diff:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  Fraction GAD < MDD: {frac_negative:.3f}")

    return {
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_diff": float(np.std(diffs)),
        "frac_gad_below_mdd": frac_negative,
        "n_trials": n_trials,
        "gad_mean": float(np.mean(gad_kappas)),
        "mdd_mean": float(np.mean(mdd_kappas)),
    }


def experiment_orc_degree_correlation():
    """ORC-degree and Forman-degree correlations across all 23 nodes."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Computing ORC-degree correlation...")
    G = build_graph()
    orc = compute_orc(G, alpha=0.5)
    mc = mean_curvature(G, orc)

    nodes = sorted(G.nodes())
    degrees = np.array([G.degree(v) for v in nodes])
    orc_vals = np.array([mc[v] for v in nodes])

    forman_vals = np.array([np.mean([4 - G.degree(u) - G.degree(v)
                                      for u in G.neighbors(v)])
                             for v in nodes])

    orc_pearson = stats.pearsonr(degrees, orc_vals)
    orc_spearman = stats.spearmanr(degrees, orc_vals)
    forman_pearson = stats.pearsonr(degrees, forman_vals)
    forman_spearman = stats.spearmanr(degrees, forman_vals)

    print(f"  ORC-degree:    Pearson r={orc_pearson[0]:.3f} (p={orc_pearson[1]:.4f}), "
          f"Spearman ρ={orc_spearman.correlation:.3f} (p={orc_spearman.pvalue:.4f})")
    print(f"  Forman-degree: Pearson r={forman_pearson[0]:.3f} (p={forman_pearson[1]:.4f}), "
          f"Spearman ρ={forman_spearman.correlation:.3f} (p={forman_spearman.pvalue:.4f})")

    return {
        "orc_degree": {
            "pearson_r": float(orc_pearson[0]),
            "pearson_p": float(orc_pearson[1]),
            "spearman_rho": float(orc_spearman.correlation),
            "spearman_p": float(orc_spearman.pvalue),
        },
        "forman_degree": {
            "pearson_r": float(forman_pearson[0]),
            "pearson_p": float(forman_pearson[1]),
            "spearman_rho": float(forman_spearman.correlation),
            "spearman_p": float(forman_spearman.pvalue),
        },
        "per_node": [
            {"node": v, "degree": int(G.degree(v)),
             "orc_kappa": float(mc[v]), "forman_mean": float(forman_vals[i])}
            for i, v in enumerate(nodes)
        ],
    }


def experiment_modularity():
    """Modularity and coverage for Ricci-flow and Louvain partitions."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Computing modularity scores...")
    G = build_graph()

    louvain_part = community_louvain.best_partition(G, random_state=42)
    louvain_mod = community_louvain.modularity(louvain_part, G)

    orc = compute_orc(G, alpha=0.5)
    H = G.copy()
    for u, v in H.edges():
        H[u][v]['weight'] = 1.0

    tau = 0.3
    threshold = 20.0
    for iteration in range(80):
        cur_orc = compute_orc(H, alpha=0.5)
        edges_to_remove = []
        for u, v in list(H.edges()):
            k = cur_orc.get((u, v), cur_orc.get((v, u), 0))
            w = H[u][v].get('weight', 1.0)
            new_w = w * (1 - tau * k)
            if new_w > threshold:
                edges_to_remove.append((u, v))
            else:
                H[u][v]['weight'] = new_w
        for u, v in edges_to_remove:
            H.remove_edge(u, v)
        if not list(H.edges()):
            break

    components = list(nx.connected_components(H))
    ricci_part = {}
    for i, comp in enumerate(components):
        for v in comp:
            ricci_part[v] = i

    ricci_mod = community_louvain.modularity(ricci_part, G)

    n_within_louvain = sum(1 for u, v in G.edges() if louvain_part[u] == louvain_part[v])
    n_within_ricci = sum(1 for u, v in G.edges() if ricci_part[u] == ricci_part[v])
    coverage_louvain = n_within_louvain / G.number_of_edges()
    coverage_ricci = n_within_ricci / G.number_of_edges()

    print(f"  Louvain: modularity={louvain_mod:.3f}, coverage={coverage_louvain:.3f}")
    print(f"  Ricci:   modularity={ricci_mod:.3f}, coverage={coverage_ricci:.3f}")

    return {
        "louvain": {
            "modularity": float(louvain_mod),
            "coverage": float(coverage_louvain),
            "n_communities": len(set(louvain_part.values())),
        },
        "ricci_flow": {
            "modularity": float(ricci_mod),
            "coverage": float(coverage_ricci),
            "n_communities": len(components),
        },
    }


def experiment_two_sided_nulls():
    """Re-report null model p-values two-sided."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Computing two-sided null model p-values...")
    r = json.load(open('results/reviewer/reviewer_results.json'))
    nm = r['null_models']

    er_z = nm['er']['z']
    cm_z = nm['cm']['z']
    er_p_two = float(2 * (1 - stats.norm.cdf(abs(er_z))))
    cm_p_two = float(2 * (1 - stats.norm.cdf(abs(cm_z))))

    print(f"  ER: z={er_z:.2f}, one-sided p={nm['er']['p']:.4f}, two-sided p={er_p_two:.4f}")
    print(f"  CM: z={cm_z:.2f}, one-sided p={nm['cm']['p']:.4f}, two-sided p={cm_p_two:.4f}")

    return {
        "er": {"z": er_z, "p_one_sided": nm['er']['p'], "p_two_sided": er_p_two},
        "cm": {"z": cm_z, "p_one_sided": nm['cm']['p'], "p_two_sided": cm_p_two},
    }


if __name__ == "__main__":
    results = {}
    results["paired_bootstrap"] = experiment_paired_bootstrap(n_trials=1000)
    results["orc_degree_correlation"] = experiment_orc_degree_correlation()
    results["modularity"] = experiment_modularity()
    results["two_sided_nulls"] = experiment_two_sided_nulls()

    out = Path("results/reviewer/v5a_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Saved to {out}")
