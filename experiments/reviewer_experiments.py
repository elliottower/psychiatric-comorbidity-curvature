"""Reviewer-requested experiments for comorbidity curvature paper V5.

Runs locally on CPU (23-node graph = instant).
Produces results/reviewer/reviewer_results.json + figures/

Experiments:
1. Comparison metrics: Forman-Ricci, local clustering coeff, edge betweenness
2. Bootstrap CIs on mean curvature from 1000 edge-perturbation trials
3. Per-node curvature distributions (mean, std, min, max, all edges)
4. Degree-2 control: are all degree-2 cross-community nodes "hidden bridges"?
5. Ricci flow sensitivity: tau x threshold grid
6. 10,000-sample null models (ER + config)
7. Figures: network viz, slope graph
"""
import json
import numpy as np
import networkx as nx
import ot
import community as community_louvain
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

EDGES = [
    ("MDD", "GAD", "Kessler 2005; Moffitt 2007"),
    ("MDD", "insomnia", "Baglioni 2011"),
    ("MDD", "SUD", "Davis 2008"),
    ("MDD", "anhedonia", "Pizzagalli 2014"),
    ("MDD", "suicide_ideation", "Nock 2008"),
    ("MDD", "PTSD", "Flory 2015"),
    ("MDD", "SCZ", "Buckley 2009"),
    ("MDD", "BIP", "Hirschfeld 2003"),
    ("MDD", "BPD", "Gunderson 2008"),
    ("MDD", "fatigue", "Demyttenaere 2005"),
    ("MDD", "cortisol", "Pariante 2008"),
    ("MDD", "inflammation", "Dowlati 2010"),
    ("MDD", "OCD", "Brakoulias 2017"),
    ("GAD", "panic", "Brown 2001"),
    ("GAD", "somatic", "Kroenke 2007"),
    ("GAD", "social_anxiety", "Mennin 2008"),
    ("GAD", "PTSD", "Ginzburg 2010"),
    ("GAD", "ADHD", "Kessler 2006"),
    ("GAD", "ASD", "van Steensel 2011"),
    ("GAD", "OCD", "Abramowitz 2003"),
    ("insomnia", "fatigue", "Lichstein 1997"),
    ("insomnia", "cortisol", "Vgontzas 2001"),
    ("panic", "agoraphobia", "APA DSM-5 2013"),
    ("SUD", "psychosis", "Niemi-Pynttari 2013"),
    ("SUD", "suicide_ideation", "Wilcox 2004"),
    ("SUD", "PTSD", "Jacobsen 2001"),
    ("SUD", "BIP", "Regier 1990"),
    ("SUD", "ADHD", "Lee 2011"),
    ("SUD", "anhedonia", "Garfield 2014"),
    ("SUD", "BPD", "Trull 2000"),
    ("SUD", "inflammation", "Crews 2006"),
    ("PTSD", "dissociation", "Lanius 2010"),
    ("PTSD", "cortisol", "Yehuda 2006"),
    ("SCZ", "psychosis", "APA DSM-5 2013"),
    ("ADHD", "ASD", "Rommelse 2010"),
    ("dissociation", "BPD", "Zanarini 2000"),
    ("BPD", "suicide_ideation", "Soloff 2000"),
    ("fatigue", "inflammation", "Bower 2009"),
    ("social_anxiety", "avoidance", "Hofmann 2007"),
]

def build_graph():
    G = nx.Graph()
    for a, b, _ in EDGES:
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

def forman_ricci(G):
    """Forman-Ricci curvature: for edge (u,v), F = 4 - deg(u) - deg(v)."""
    curvatures = {}
    for u, v in G.edges():
        f = 4 - G.degree(u) - G.degree(v)
        curvatures[(u, v)] = f
        curvatures[(v, u)] = f
    return curvatures

def mean_forman(G, curvatures):
    result = {}
    for v in G.nodes():
        edges = [(v, u) for u in G.neighbors(v)]
        result[v] = np.mean([curvatures[(a, b)] for a, b in edges])
    return result

def edge_betweenness_node(G):
    """Mean edge betweenness of edges incident to each node."""
    eb = nx.edge_betweenness_centrality(G)
    result = {}
    for v in G.nodes():
        edges = []
        for u in G.neighbors(v):
            key = (v, u) if (v, u) in eb else (u, v)
            edges.append(eb[key])
        result[v] = np.mean(edges)
    return result

def rank_nodes(scores, ascending=True):
    """Rank nodes by score. ascending=True means lowest score = rank 1 (most bridge-like)."""
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=not ascending)
    return {node: rank + 1 for rank, (node, _) in enumerate(sorted_nodes)}

def perturb_graph(G, rng):
    """Remove 1-20% edges, add 0-10% absent edges, preserving connectivity."""
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

# ============================================================
# EXPERIMENT 1: Comparison metrics
# ============================================================
def experiment_comparison_metrics():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running comparison metrics...")
    G = build_graph()
    orc = compute_orc(G, alpha=0.5)
    mc = mean_curvature(G, orc)
    fr = forman_ricci(G)
    mfr = mean_forman(G, fr)
    cc = nx.clustering(G)
    btwn = nx.betweenness_centrality(G)
    eb = edge_betweenness_node(G)

    orc_rank = rank_nodes(mc, ascending=True)
    fr_rank = rank_nodes(mfr, ascending=True)
    cc_rank = rank_nodes(cc, ascending=True)
    btwn_rank = rank_nodes(btwn, ascending=False)
    eb_rank = rank_nodes(eb, ascending=False)

    rows = []
    for node in sorted(G.nodes()):
        rows.append({
            "node": node,
            "degree": G.degree(node),
            "betweenness": round(btwn[node], 4),
            "btwn_rank": btwn_rank[node],
            "orc_mean": round(mc[node], 4),
            "orc_rank": orc_rank[node],
            "forman_mean": round(mfr[node], 2),
            "forman_rank": fr_rank[node],
            "clustering": round(cc[node], 4),
            "cc_rank": cc_rank[node],
            "edge_btwn_mean": round(eb[node], 4),
            "eb_rank": eb_rank[node],
        })

    top_bridge_by = {
        "ORC": sorted(mc.items(), key=lambda x: x[1])[0][0],
        "Forman": sorted(mfr.items(), key=lambda x: x[1])[0][0],
        "clustering_coeff": sorted(cc.items(), key=lambda x: x[1])[0][0],
        "edge_betweenness": sorted(eb.items(), key=lambda x: x[1], reverse=True)[0][0],
        "betweenness": sorted(btwn.items(), key=lambda x: x[1], reverse=True)[0][0],
    }

    orc_top3 = [n for n, _ in sorted(mc.items(), key=lambda x: x[1])[:3]]
    fr_top3 = [n for n, _ in sorted(mfr.items(), key=lambda x: x[1])[:3]]
    cc_bottom3 = [n for n, _ in sorted(cc.items(), key=lambda x: x[1])[:3]]
    eb_top3 = [n for n, _ in sorted(eb.items(), key=lambda x: x[1], reverse=True)[:3]]

    gad_hub_bridge = {
        "ORC": {"rank": orc_rank["GAD"], "value": round(mc["GAD"], 4), "distinguishes": mc["GAD"] < mc["MDD"]},
        "Forman": {"rank": fr_rank["GAD"], "value": round(mfr["GAD"], 2), "distinguishes": mfr["GAD"] < mfr["MDD"]},
        "clustering": {"rank": cc_rank["GAD"], "value": round(cc["GAD"], 4), "distinguishes": cc["GAD"] < cc["MDD"]},
        "edge_betweenness": {"rank": eb_rank["GAD"], "value": round(eb["GAD"], 4), "distinguishes": eb["GAD"] > eb["MDD"]},
    }
    mdd_vals = {
        "ORC": round(mc["MDD"], 4),
        "Forman": round(mfr["MDD"], 2),
        "clustering": round(cc["MDD"], 4),
        "edge_betweenness": round(eb["MDD"], 4),
    }

    return {
        "per_node": rows,
        "top_bridge_by_metric": top_bridge_by,
        "orc_top3": orc_top3,
        "forman_top3": fr_top3,
        "cc_bottom3": cc_bottom3,
        "eb_top3": eb_top3,
        "gad_hub_bridge_test": gad_hub_bridge,
        "mdd_values": mdd_vals,
    }

# ============================================================
# EXPERIMENT 2: Bootstrap CIs from 1000 perturbation trials
# ============================================================
def experiment_bootstrap_ci():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running bootstrap CIs (1000 trials)...")
    G = build_graph()
    rng = np.random.default_rng(42)
    n_trials = 1000
    node_curvatures = defaultdict(list)
    node_ranks = defaultdict(list)

    for t in range(n_trials):
        if t % 200 == 0:
            print(f"  trial {t}/{n_trials}")
        H = perturb_graph(G, rng)
        orc = compute_orc(H, alpha=0.5)
        mc = mean_curvature(H, orc)
        rank = rank_nodes(mc, ascending=True)
        for node in H.nodes():
            node_curvatures[node].append(mc[node])
            node_ranks[node].append(rank[node])

    results = {}
    for node in sorted(G.nodes()):
        vals = node_curvatures[node]
        ranks = node_ranks[node]
        results[node] = {
            "mean_kappa": round(np.mean(vals), 4),
            "ci_lower": round(np.percentile(vals, 2.5), 4),
            "ci_upper": round(np.percentile(vals, 97.5), 4),
            "std": round(np.std(vals), 4),
            "mean_rank": round(np.mean(ranks), 1),
            "rank_ci_lower": round(np.percentile(ranks, 2.5), 1),
            "rank_ci_upper": round(np.percentile(ranks, 97.5), 1),
            "rank1_frac": round(sum(1 for r in ranks if r == 1) / len(ranks), 3),
            "top3_frac": round(sum(1 for r in ranks if r <= 3) / len(ranks), 3),
        }
    return results

# ============================================================
# EXPERIMENT 3: Per-node curvature distributions
# ============================================================
def experiment_curvature_distributions():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Computing per-node curvature distributions...")
    G = build_graph()
    orc = compute_orc(G, alpha=0.5)
    results = {}
    for v in sorted(G.nodes()):
        edge_curvs = []
        for u in G.neighbors(v):
            edge_curvs.append({"neighbor": u, "kappa": round(orc[(v, u)], 4)})
        vals = [e["kappa"] for e in edge_curvs]
        n_neg = sum(1 for x in vals if x < 0)
        n_pos = sum(1 for x in vals if x > 0)
        results[v] = {
            "degree": G.degree(v),
            "edges": edge_curvs,
            "mean": round(np.mean(vals), 4),
            "std": round(np.std(vals), 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "n_negative": n_neg,
            "n_positive": n_pos,
            "fraction_negative": round(n_neg / len(vals), 3),
        }
    return results

# ============================================================
# EXPERIMENT 4: Degree-2 control for OCD
# ============================================================
def experiment_degree2_control():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running degree-2 control...")
    G = build_graph()
    orc = compute_orc(G, alpha=0.5)
    mc = mean_curvature(G, orc)
    btwn = nx.betweenness_centrality(G)
    orc_rank = rank_nodes(mc, ascending=True)
    btwn_rank = rank_nodes(btwn, ascending=False)

    ricci_communities = {
        0: ["MDD", "BIP", "SCZ", "PTSD", "SUD", "BPD", "anhedonia", "psychosis",
            "dissociation", "insomnia", "fatigue", "inflammation", "cortisol", "suicide_ideation"],
        1: ["ADHD", "ASD", "GAD", "OCD", "somatic"],
        2: ["panic", "agoraphobia"],
        3: ["social_anxiety", "avoidance"],
    }
    node_to_comm = {}
    for comm, members in ricci_communities.items():
        for m in members:
            node_to_comm[m] = comm

    degree2_nodes = [v for v in G.nodes() if G.degree(v) == 2]
    degree2_analysis = []
    for v in degree2_nodes:
        nbrs = list(G.neighbors(v))
        nbr_comms = [node_to_comm.get(u, -1) for u in nbrs]
        cross_community = len(set(nbr_comms)) > 1
        degree2_analysis.append({
            "node": v,
            "neighbors": nbrs,
            "neighbor_communities": nbr_comms,
            "cross_community": cross_community,
            "mean_kappa": round(mc[v], 4),
            "orc_rank": orc_rank[v],
            "btwn_rank": btwn_rank[v],
            "rank_change": btwn_rank[v] - orc_rank[v],
            "betweenness": round(btwn[v], 4),
        })

    cross_comm_d2 = [d for d in degree2_analysis if d["cross_community"]]
    same_comm_d2 = [d for d in degree2_analysis if not d["cross_community"]]

    return {
        "degree2_nodes": degree2_analysis,
        "n_degree2": len(degree2_nodes),
        "n_cross_community": len(cross_comm_d2),
        "n_same_community": len(same_comm_d2),
        "cross_comm_mean_rank_change": round(np.mean([d["rank_change"] for d in cross_comm_d2]), 1) if cross_comm_d2 else None,
        "same_comm_mean_rank_change": round(np.mean([d["rank_change"] for d in same_comm_d2]), 1) if same_comm_d2 else None,
        "conclusion": "OCD rank jump is specific to cross-community degree-2 nodes" if (
            cross_comm_d2 and same_comm_d2 and
            np.mean([d["rank_change"] for d in cross_comm_d2]) > np.mean([d["rank_change"] for d in same_comm_d2]) + 3
        ) else "OCD rank jump is partly a degree-2 topological effect",
    }

# ============================================================
# EXPERIMENT 5: Ricci flow sensitivity
# ============================================================
def ricci_flow_communities(G, tau, threshold, max_iter=80):
    H = G.copy()
    for u, v in H.edges():
        H[u][v]['weight'] = 1.0
    for it in range(max_iter):
        orc = compute_orc_weighted(H, alpha=0.5)
        changed = False
        for u, v in list(H.edges()):
            k = orc.get((u, v), 0)
            w = H[u][v]['weight']
            new_w = w * (1 - tau * k)
            H[u][v]['weight'] = new_w
            if new_w > threshold:
                H.remove_edge(u, v)
                changed = True
        if not changed and it > 10:
            break
    comms = list(nx.connected_components(H))
    return len(comms), [sorted(list(c)) for c in comms]

def compute_orc_weighted(G, alpha=0.5):
    nodes = sorted(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
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
        total_w = sum(1.0 / G[u][w].get('weight', 1.0) for w in nbrs_u)
        for w in nbrs_u:
            mu_u[idx[w]] = (1 - alpha) * (1.0 / G[u][w].get('weight', 1.0)) / total_w
        mu_v = np.zeros(n)
        mu_v[idx[v]] = alpha
        nbrs_v = list(G.neighbors(v))
        total_w = sum(1.0 / G[v][w].get('weight', 1.0) for w in nbrs_v)
        for w in nbrs_v:
            mu_v[idx[w]] = (1 - alpha) * (1.0 / G[v][w].get('weight', 1.0)) / total_w
        d_uv = D[idx[u], idx[v]]
        if d_uv == 0:
            continue
        w1 = ot.emd2(mu_u, mu_v, D)
        kappa = 1 - w1 / d_uv
        curvatures[(u, v)] = kappa
        curvatures[(v, u)] = kappa
    return curvatures

def experiment_ricci_flow_sensitivity():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running Ricci flow sensitivity...")
    G = build_graph()
    taus = [0.1, 0.2, 0.3, 0.5]
    thresholds = [10, 15, 20, 30]
    results = []
    for tau in taus:
        for thresh in thresholds:
            print(f"  tau={tau}, threshold={thresh}")
            n_comm, comms = ricci_flow_communities(G, tau, thresh)
            sizes = sorted([len(c) for c in comms], reverse=True)
            results.append({
                "tau": tau,
                "threshold": thresh,
                "n_communities": n_comm,
                "community_sizes": sizes,
            })
    return results

# ============================================================
# EXPERIMENT 6: 10,000-sample null models
# ============================================================
def experiment_null_models():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running 10,000-sample null models...")
    G = build_graph()
    orc = compute_orc(G, alpha=0.5)
    real_mean = np.mean(list(set(orc.values())))

    n_samples = 10000
    rng = np.random.default_rng(123)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    p = 2 * m / (n * (n - 1))
    deg_seq = sorted([d for _, d in G.degree()], reverse=True)

    er_means = []
    cm_means = []
    for i in range(n_samples):
        if i % 2000 == 0:
            print(f"  null sample {i}/{n_samples}")
        # ER null
        H_er = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(0, 2**31)))
        if H_er.number_of_edges() > 0 and nx.is_connected(H_er):
            orc_er = compute_orc(H_er, alpha=0.5)
            er_means.append(np.mean(list(set(orc_er.values()))))
        # Config model null
        try:
            H_cm = nx.configuration_model(deg_seq, seed=int(rng.integers(0, 2**31)))
            H_cm = nx.Graph(H_cm)
            H_cm.remove_edges_from(nx.selfloop_edges(H_cm))
            if H_cm.number_of_edges() > 0 and nx.is_connected(H_cm):
                orc_cm = compute_orc(H_cm, alpha=0.5)
                cm_means.append(np.mean(list(set(orc_cm.values()))))
        except Exception:
            pass

    er_z = (real_mean - np.mean(er_means)) / np.std(er_means) if er_means else None
    cm_z = (real_mean - np.mean(cm_means)) / np.std(cm_means) if cm_means else None

    return {
        "real_mean_curvature": round(real_mean, 4),
        "er": {
            "n_valid": len(er_means),
            "mean": round(np.mean(er_means), 4) if er_means else None,
            "std": round(np.std(er_means), 4) if er_means else None,
            "z": round(er_z, 2) if er_z else None,
            "p": round(sum(1 for x in er_means if x >= real_mean) / len(er_means), 4) if er_means else None,
        },
        "cm": {
            "n_valid": len(cm_means),
            "mean": round(np.mean(cm_means), 4) if cm_means else None,
            "std": round(np.std(cm_means), 4) if cm_means else None,
            "z": round(cm_z, 2) if cm_z else None,
            "p": round(sum(1 for x in cm_means if x >= real_mean) / len(cm_means), 4) if cm_means else None,
        },
    }

# ============================================================
# EXPERIMENT 7: Figures
# ============================================================
def generate_figures(comparison_results, bootstrap_results, curvature_dists):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating figures...")
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    G = build_graph()
    orc = compute_orc(G, alpha=0.5)
    mc = mean_curvature(G, orc)
    btwn = nx.betweenness_centrality(G)

    # --- Figure 1: Network visualization ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=1.5)

    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        k = orc.get((u, v), orc.get((v, u), 0))
        if k < -0.2:
            edge_colors.append('#d62728')
            edge_widths.append(2.5)
        elif k < 0:
            edge_colors.append('#ff7f0e')
            edge_widths.append(1.8)
        elif k < 0.2:
            edge_colors.append('#7f7f7f')
            edge_widths.append(1.0)
        else:
            edge_colors.append('#2ca02c')
            edge_widths.append(1.5)

    nx.draw_edges = nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths, alpha=0.7)

    node_colors = []
    for v in G.nodes():
        k = mc[v]
        if k < -0.05:
            node_colors.append('#d62728')
        elif k < 0.05:
            node_colors.append('#ff7f0e')
        elif k < 0.15:
            node_colors.append('#7f7f7f')
        else:
            node_colors.append('#2ca02c')

    node_sizes = [300 + G.degree(v) * 120 for v in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
                           edgecolors='black', linewidths=1.0)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_weight='bold')

    ax.set_title('Comorbidity network colored by mean incident curvature\n'
                 'Red: bridge ($\\bar{\\kappa} < -0.05$)  Orange: mixed  '
                 'Gray: neutral  Green: cluster ($\\bar{\\kappa} > 0.15$)', fontsize=11)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig_network.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(fig_dir / 'fig_network.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Figure 2: Slope graph (betweenness rank -> curvature rank) ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    orc_rank = rank_nodes(mc, ascending=True)
    btwn_rank = rank_nodes(btwn, ascending=False)
    n_nodes = len(G.nodes())

    highlight = {"OCD": '#d62728', "GAD": '#1f77b4', "MDD": '#ff7f0e',
                 "somatic": '#9467bd', "BIP": '#8c564b', "insomnia": '#e377c2'}

    for v in G.nodes():
        br = btwn_rank[v]
        cr = orc_rank[v]
        delta = br - cr
        color = highlight.get(v, '#cccccc')
        lw = 2.0 if v in highlight else 0.5
        alpha = 1.0 if v in highlight else 0.3
        ax.plot([0, 1], [br, cr], color=color, linewidth=lw, alpha=alpha)
        if v in highlight:
            ax.annotate(f'{v}', xy=(0, br), fontsize=8, ha='right', va='center',
                        fontweight='bold', color=color)
            label = f'{v} ({delta:+d})' if abs(delta) >= 3 else v
            ax.annotate(label, xy=(1, cr), fontsize=8, ha='left', va='center',
                        fontweight='bold', color=color)

    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(n_nodes + 0.5, 0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Betweenness\nrank', 'Curvature\nbridge rank'], fontsize=10)
    ax.set_ylabel('Rank (1 = most important)', fontsize=10)
    ax.set_title('Rank changes: betweenness vs. curvature', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig_slopegraph.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(fig_dir / 'fig_slopegraph.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Figure 3: Metric comparison (which metrics distinguish hub from bridge?) ---
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    metrics = [
        ("ORC $\\bar{\\kappa}$", {r["node"]: r["orc_mean"] for r in comparison_results["per_node"]}, True),
        ("Forman $\\bar{F}$", {r["node"]: r["forman_mean"] for r in comparison_results["per_node"]}, True),
        ("Clustering coeff.", {r["node"]: r["clustering"] for r in comparison_results["per_node"]}, True),
        ("Edge betweenness", {r["node"]: r["edge_btwn_mean"] for r in comparison_results["per_node"]}, False),
    ]
    for ax_i, (title, vals, lower_is_bridge) in zip(axes, metrics):
        gad_val = vals["GAD"]
        mdd_val = vals["MDD"]
        distinguishes = (gad_val < mdd_val) if lower_is_bridge else (gad_val > mdd_val)

        all_vals = sorted(vals.values())
        for v, val in vals.items():
            color = '#1f77b4' if v == 'GAD' else '#ff7f0e' if v == 'MDD' else '#cccccc'
            ms = 10 if v in ('GAD', 'MDD') else 4
            ax_i.plot(val, 0, 'o', color=color, markersize=ms, zorder=3 if v in ('GAD', 'MDD') else 1)

        ax_i.set_title(title, fontsize=10)
        ax_i.set_yticks([])
        status = '(distinguishes)' if distinguishes else '(fails)'
        ax_i.set_xlabel(status, fontsize=9,
                        color='green' if distinguishes else 'red')
        ax_i.spines['top'].set_visible(False)
        ax_i.spines['right'].set_visible(False)
        ax_i.spines['left'].set_visible(False)

    fig.suptitle('Which metrics distinguish GAD (blue) from MDD (orange)?', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig_metric_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(fig_dir / 'fig_metric_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Figure 4: Bootstrap CI forest plot ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    nodes_sorted = sorted(bootstrap_results.keys(),
                          key=lambda v: bootstrap_results[v]["mean_kappa"])
    y_pos = range(len(nodes_sorted))
    for i, v in enumerate(nodes_sorted):
        d = bootstrap_results[v]
        color = '#d62728' if d["mean_kappa"] < -0.05 else '#ff7f0e' if d["mean_kappa"] < 0.05 else '#2ca02c'
        ax.errorbar(d["mean_kappa"], i,
                     xerr=[[d["mean_kappa"] - d["ci_lower"]], [d["ci_upper"] - d["mean_kappa"]]],
                     fmt='o', color=color, capsize=3, markersize=6)
    ax.axvline(x=0, color='black', linewidth=0.5, linestyle='--')
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(nodes_sorted, fontsize=8)
    ax.set_xlabel('Mean incident curvature $\\bar{\\kappa}$ (95% CI from 1000 perturbation trials)', fontsize=10)
    ax.set_title('Bootstrap confidence intervals on $\\bar{\\kappa}$', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig_bootstrap_ci.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(fig_dir / 'fig_bootstrap_ci.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Figures saved to {fig_dir}/")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    out_dir = Path("results/reviewer")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    results["comparison_metrics"] = experiment_comparison_metrics()
    results["curvature_distributions"] = experiment_curvature_distributions()
    results["degree2_control"] = experiment_degree2_control()
    results["ricci_flow_sensitivity"] = experiment_ricci_flow_sensitivity()
    results["bootstrap_ci"] = experiment_bootstrap_ci()
    results["null_models"] = experiment_null_models()

    generate_figures(results["comparison_metrics"], results["bootstrap_ci"],
                     results["curvature_distributions"])

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return super().default(obj)

    out_path = out_dir / "reviewer_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] All results saved to {out_path}")
