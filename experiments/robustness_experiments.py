"""Robustness experiments for comorbidity curvature paper.

Runs locally on CPU (23-node graph = instant).
Produces results/robustness/robustness_results.json

Experiments:
1. Alpha sweep (0.1 to 0.9)
2. Edge perturbation sensitivity (100 trials, 10-20% perturbation)
3. Ricci flow convergence trajectory
4. Louvain/spectral community comparison
5. Clinical novelty: curvature reranking vs betweenness
"""
import json
import numpy as np
import networkx as nx
import ot
import community as community_louvain
from pathlib import Path
from collections import Counter

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
    for u, v, cite in EDGES:
        G.add_edge(u, v, citation=cite)
    return G


def compute_orc(G, alpha=0.5):
    sp = dict(nx.all_pairs_shortest_path_length(G))
    curvatures = {}
    for u, v in G.edges():
        nb_u = list(G.neighbors(u))
        nb_v = list(G.neighbors(v))
        all_nodes = sorted(set([u] + nb_u + [v] + nb_v))
        idx = {nd: i for i, nd in enumerate(all_nodes)}

        mu = np.zeros(len(all_nodes))
        mu[idx[u]] = alpha
        for nb in nb_u:
            mu[idx[nb]] += (1 - alpha) / len(nb_u)

        nu = np.zeros(len(all_nodes))
        nu[idx[v]] = alpha
        for nb in nb_v:
            nu[idx[nb]] += (1 - alpha) / len(nb_v)

        cost = np.zeros((len(all_nodes), len(all_nodes)))
        for ci, ni in enumerate(all_nodes):
            for cj, nj in enumerate(all_nodes):
                cost[ci, cj] = sp.get(ni, {}).get(nj, 100)

        W1 = ot.emd2(mu, nu, cost)
        d_uv = sp[u][v]
        curvatures[(u, v)] = float(1.0 - W1 / d_uv) if d_uv > 0 else 0.0
    return curvatures


def node_mean_curvature(G, curvatures):
    result = {}
    for node in G.nodes():
        incident = []
        for (u, v), k in curvatures.items():
            if u == node or v == node:
                incident.append(k)
        result[node] = float(np.mean(incident)) if incident else 0.0
    return result


def get_bridge_ranking(G, curvatures):
    nmc = node_mean_curvature(G, curvatures)
    return sorted(nmc.items(), key=lambda x: x[1])


# ============================================================
# Experiment 1: Alpha sweep
# ============================================================
def run_alpha_sweep(G):
    print("=== Alpha sweep ===")
    results = {}
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        curv = compute_orc(G, alpha=alpha)
        nmc = node_mean_curvature(G, curv)
        ranking = sorted(nmc.items(), key=lambda x: x[1])
        top_bridge = ranking[0][0]
        top3 = [r[0] for r in ranking[:3]]

        most_neg_edges = sorted(curv.items(), key=lambda x: x[1])[:3]
        most_neg_edge_strs = [f"{u}--{v}: {k:.3f}" for (u, v), k in most_neg_edges]

        results[str(alpha)] = {
            "top_bridge": top_bridge,
            "top3_bridges": top3,
            "top_bridge_curvature": float(ranking[0][1]),
            "mean_curvature": float(np.mean(list(curv.values()))),
            "most_negative_edges": most_neg_edge_strs,
        }
        print(f"  α={alpha:.1f}: top bridge={top_bridge} "
              f"(κ_avg={ranking[0][1]:+.3f}), "
              f"mean κ={np.mean(list(curv.values())):+.4f}")
    return results


# ============================================================
# Experiment 2: Edge perturbation sensitivity
# ============================================================
def run_edge_perturbation(G, n_trials=200):
    print("\n=== Edge perturbation (200 trials) ===")
    rng = np.random.default_rng(42)
    all_nodes = sorted(G.nodes())
    edges = list(G.edges())
    n_edges = len(edges)

    possible_edges = []
    for i, u in enumerate(all_nodes):
        for v in all_nodes[i+1:]:
            if not G.has_edge(u, v):
                possible_edges.append((u, v))

    gad_rank1_count = 0
    gad_top3_count = 0
    top_bridge_counts = Counter()

    for trial in range(n_trials):
        G_pert = G.copy()
        n_remove = rng.integers(1, max(2, int(0.2 * n_edges)) + 1)
        n_add = rng.integers(0, max(1, int(0.1 * n_edges)) + 1)

        remove_idx = rng.choice(n_edges, size=min(n_remove, n_edges - 1), replace=False)
        for idx in remove_idx:
            u, v = edges[idx]
            if G_pert.has_edge(u, v) and G_pert.degree(u) > 1 and G_pert.degree(v) > 1:
                G_pert.remove_edge(u, v)

        if possible_edges and n_add > 0:
            add_idx = rng.choice(len(possible_edges), size=min(n_add, len(possible_edges)), replace=False)
            for idx in add_idx:
                u, v = possible_edges[idx]
                G_pert.add_edge(u, v)

        if G_pert.number_of_edges() < 5:
            continue

        curv = compute_orc(G_pert)
        ranking = get_bridge_ranking(G_pert, curv)
        top = ranking[0][0]
        top3 = [r[0] for r in ranking[:3]]

        top_bridge_counts[top] += 1
        if top == "GAD":
            gad_rank1_count += 1
        if "GAD" in top3:
            gad_top3_count += 1

    total = sum(top_bridge_counts.values())
    results = {
        "n_trials": total,
        "gad_rank1_fraction": gad_rank1_count / total if total > 0 else 0,
        "gad_top3_fraction": gad_top3_count / total if total > 0 else 0,
        "top_bridge_distribution": {k: v / total for k, v in top_bridge_counts.most_common()},
    }
    print(f"  GAD rank #1: {gad_rank1_count}/{total} "
          f"({100 * results['gad_rank1_fraction']:.1f}%)")
    print(f"  GAD in top 3: {gad_top3_count}/{total} "
          f"({100 * results['gad_top3_fraction']:.1f}%)")
    print(f"  Top bridge distribution: "
          f"{dict(top_bridge_counts.most_common(5))}")
    return results


# ============================================================
# Experiment 3: Ricci flow convergence
# ============================================================
def run_ricci_flow_convergence(G):
    print("\n=== Ricci flow convergence ===")
    G_flow = G.copy()
    trajectory = []
    n_iters = 80
    tau = 0.3

    for it in range(n_iters):
        curv = compute_orc(G_flow)
        vals = list(curv.values())
        n_comp = nx.number_connected_components(G_flow)
        trajectory.append({
            "iteration": it,
            "mean_curvature": float(np.mean(vals)),
            "std_curvature": float(np.std(vals)),
            "n_edges": G_flow.number_of_edges(),
            "n_components": n_comp,
            "n_negative": int(np.sum(np.array(vals) < -0.01)),
        })

        edges_to_remove = []
        for (u, v), k in curv.items():
            old_w = G_flow[u][v].get("weight", 1.0)
            new_w = old_w * (1.0 - tau * k)
            new_w = max(new_w, 0.01)
            if new_w > 20.0:
                edges_to_remove.append((u, v))
            else:
                G_flow[u][v]["weight"] = new_w

        for u, v in edges_to_remove:
            G_flow.remove_edge(u, v)

        if it % 10 == 0:
            print(f"  iter {it:3d}: mean_κ={np.mean(vals):+.4f} "
                  f"edges={G_flow.number_of_edges()} components={n_comp}")

        if n_comp >= 4 and G_flow.number_of_edges() < 30:
            # Record remaining trajectory as stable
            for it2 in range(it + 1, n_iters):
                trajectory.append(trajectory[-1].copy())
                trajectory[-1]["iteration"] = it2
            break

    components = list(nx.connected_components(G_flow))
    communities = {}
    for i, comp in enumerate(sorted(components, key=len, reverse=True)):
        communities[f"community_{i}"] = sorted(comp)

    # At what iteration did we first get >1 component?
    first_split = None
    for t in trajectory:
        if t["n_components"] > 1:
            first_split = t["iteration"]
            break

    results = {
        "trajectory": trajectory,
        "final_communities": communities,
        "first_split_iteration": first_split,
        "converged_by": trajectory[-1]["iteration"],
    }
    print(f"  First split at iteration {first_split}")
    print(f"  Final communities: {communities}")
    return results


# ============================================================
# Experiment 4: Louvain / spectral comparison
# ============================================================
def run_community_comparison(G, ricci_communities):
    print("\n=== Community detection comparison ===")

    # Louvain
    louvain_partition = community_louvain.best_partition(G, random_state=42)
    louvain_comms = {}
    for node, comm_id in louvain_partition.items():
        louvain_comms.setdefault(comm_id, []).append(node)
    louvain_comms = {f"community_{i}": sorted(v)
                     for i, v in enumerate(sorted(louvain_comms.values(),
                                                   key=len, reverse=True))}

    # Spectral (using Fiedler vector for bisection, then recursive)
    L = nx.laplacian_matrix(G).toarray().astype(float)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    fiedler = eigenvectors[:, 1]
    nodes = sorted(G.nodes())
    spectral_partition = {}
    for i, node in enumerate(nodes):
        spectral_partition[node] = 0 if fiedler[i] < 0 else 1
    spectral_comms = {}
    for node, comm_id in spectral_partition.items():
        spectral_comms.setdefault(comm_id, []).append(node)
    spectral_comms = {f"community_{i}": sorted(v)
                      for i, v in enumerate(sorted(spectral_comms.values(),
                                                    key=len, reverse=True))}

    # Normalized Mutual Information between partitions
    def partition_to_labels(partition, nodes):
        labels = []
        for node in nodes:
            for comm_name, members in partition.items():
                if node in members:
                    labels.append(comm_name)
                    break
        return labels

    def nmi(labels_a, labels_b):
        from collections import Counter
        n = len(labels_a)
        counts_a = Counter(labels_a)
        counts_b = Counter(labels_b)
        joint = Counter(zip(labels_a, labels_b))

        mi = 0.0
        for (a, b), n_ab in joint.items():
            if n_ab > 0:
                mi += (n_ab / n) * np.log2((n * n_ab) / (counts_a[a] * counts_b[b]))

        def entropy(counts):
            return -sum((c / n) * np.log2(c / n) for c in counts.values() if c > 0)

        h_a = entropy(counts_a)
        h_b = entropy(counts_b)
        if h_a + h_b == 0:
            return 1.0
        return 2 * mi / (h_a + h_b)

    nodes_list = sorted(G.nodes())
    ricci_labels = partition_to_labels(ricci_communities, nodes_list)
    louvain_labels = partition_to_labels(louvain_comms, nodes_list)
    spectral_labels = partition_to_labels(spectral_comms, nodes_list)

    nmi_ricci_louvain = nmi(ricci_labels, louvain_labels)
    nmi_ricci_spectral = nmi(ricci_labels, spectral_labels)
    nmi_louvain_spectral = nmi(louvain_labels, spectral_labels)

    results = {
        "ricci_communities": ricci_communities,
        "louvain_communities": louvain_comms,
        "spectral_communities": spectral_comms,
        "n_ricci": len(ricci_communities),
        "n_louvain": len(louvain_comms),
        "n_spectral": len(spectral_comms),
        "nmi_ricci_louvain": float(nmi_ricci_louvain),
        "nmi_ricci_spectral": float(nmi_ricci_spectral),
        "nmi_louvain_spectral": float(nmi_louvain_spectral),
    }

    print(f"  Ricci: {len(ricci_communities)} communities")
    for name, members in ricci_communities.items():
        print(f"    {name}: {members}")
    print(f"  Louvain: {len(louvain_comms)} communities")
    for name, members in louvain_comms.items():
        print(f"    {name}: {members}")
    print(f"  Spectral (bisection): {len(spectral_comms)} communities")
    for name, members in spectral_comms.items():
        print(f"    {name}: {members}")
    print(f"  NMI(Ricci, Louvain)  = {nmi_ricci_louvain:.3f}")
    print(f"  NMI(Ricci, Spectral) = {nmi_ricci_spectral:.3f}")
    print(f"  NMI(Louvain, Spectral) = {nmi_louvain_spectral:.3f}")
    return results


# ============================================================
# Experiment 5: Clinical novelty — reranking vs betweenness
# ============================================================
def run_clinical_novelty(G, curvatures):
    print("\n=== Clinical novelty: curvature vs betweenness reranking ===")
    betweenness = nx.betweenness_centrality(G)
    nmc = node_mean_curvature(G, curvatures)

    btwn_ranking = sorted(betweenness.items(), key=lambda x: -x[1])
    curv_ranking = sorted(nmc.items(), key=lambda x: x[1])

    btwn_ranks = {node: i for i, (node, _) in enumerate(btwn_ranking)}
    curv_ranks = {node: i for i, (node, _) in enumerate(curv_ranking)}

    reranking = []
    for node in sorted(G.nodes()):
        delta = btwn_ranks[node] - curv_ranks[node]
        reranking.append({
            "node": node,
            "betweenness_rank": btwn_ranks[node] + 1,
            "curvature_bridge_rank": curv_ranks[node] + 1,
            "rank_change": delta,
            "betweenness": float(betweenness[node]),
            "mean_curvature": float(nmc[node]),
            "degree": G.degree(node),
        })

    reranking.sort(key=lambda x: abs(x["rank_change"]), reverse=True)

    print(f"  {'Node':20s} {'Btwn rank':>10s} {'Curv rank':>10s} {'Δ':>5s}  "
          f"{'Btwn':>6s} {'κ_avg':>8s}")
    for r in reranking:
        print(f"  {r['node']:20s} {r['betweenness_rank']:10d} "
              f"{r['curvature_bridge_rank']:10d} {r['rank_change']:+5d}  "
              f"{r['betweenness']:6.3f} {r['mean_curvature']:+8.4f}")

    biggest_movers = [r for r in reranking if abs(r["rank_change"]) >= 3]

    results = {
        "full_reranking": reranking,
        "biggest_movers": biggest_movers,
        "gad_betweenness_rank": btwn_ranks["GAD"] + 1,
        "gad_curvature_rank": curv_ranks["GAD"] + 1,
        "mdd_betweenness_rank": btwn_ranks["MDD"] + 1,
        "mdd_curvature_rank": curv_ranks["MDD"] + 1,
    }

    print(f"\n  GAD: betweenness #{btwn_ranks['GAD']+1} → "
          f"curvature bridge #{curv_ranks['GAD']+1}")
    print(f"  MDD: betweenness #{btwn_ranks['MDD']+1} → "
          f"curvature bridge #{curv_ranks['MDD']+1}")
    if biggest_movers:
        print(f"  Biggest movers (|Δ| ≥ 3):")
        for m in biggest_movers:
            print(f"    {m['node']}: {m['rank_change']:+d} ranks")
    return results


# ============================================================
# Main
# ============================================================
def main():
    G = build_graph()
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Baseline curvatures (α=0.5)
    base_curv = compute_orc(G, alpha=0.5)

    # Run all experiments
    alpha_results = run_alpha_sweep(G)
    perturbation_results = run_edge_perturbation(G)
    flow_results = run_ricci_flow_convergence(G)
    community_results = run_community_comparison(G, flow_results["final_communities"])
    novelty_results = run_clinical_novelty(G, base_curv)

    # Edge list with citations
    edge_list = []
    for u, v, cite in EDGES:
        k = base_curv.get((u, v), base_curv.get((v, u), None))
        edge_list.append({
            "node_a": u, "node_b": v,
            "curvature": float(k) if k is not None else None,
            "citation": cite,
        })

    results = {
        "edge_list": edge_list,
        "alpha_sweep": alpha_results,
        "edge_perturbation": perturbation_results,
        "ricci_flow_convergence": flow_results,
        "community_comparison": community_results,
        "clinical_novelty": novelty_results,
    }

    out_dir = Path("results/robustness")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "robustness_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
