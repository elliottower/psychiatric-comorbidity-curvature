"""Full curvature analysis on the expanded 84-entry catalog.

Runs all analyses from the paper on the expanded graph:
  1. Graph construction and basic stats
  2. Node-level curvature + degree-preserving null
  3. Edge-level curvature + degree-preserving null (z-scores)
  4. Degree-curvature correlation (confound check)
  5. Verdict-curvature correlation (NOW POWERED at n=84)
  6. Catalog perturbation robustness
  7. Leave-one-family-out
  8. Edge betweenness comparison
  9. Comparison: original (n=47) vs expanded (n=84)

Usage:
    uv run --no-project --with numpy --with scipy --with networkx --with pot --with tqdm --with matplotlib \
        python psych/experiments/expanded_curvature_analysis.py
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
from catalog_data_expanded import ENTRIES, SHARED_NODES, FAMILIES, VERDICTS


def build_graph(entries=None, shared_nodes=None, families=None):
    if entries is None:
        entries = ENTRIES
    if shared_nodes is None:
        shared_nodes = SHARED_NODES
    if families is None:
        families = FAMILIES

    G = nx.Graph()
    for fam in families:
        G.add_node(f"family:{fam}", layer="family")

    used_mechanisms = set()
    for entry in entries:
        for node in entry["shared_nodes"]:
            used_mechanisms.add(node)

    for node_name, info in shared_nodes.items():
        if node_name in used_mechanisms:
            G.add_node(f"mechanism:{node_name}", layer="mechanism",
                       n_disorders=len(info["disorders"]))
            for disorder in info["disorders"]:
                if f"family:{disorder}" in G:
                    G.add_edge(f"family:{disorder}", f"mechanism:{node_name}")

    for entry in entries:
        eid = f"claim:{entry['id']}"
        verdict = entry["verdict"]
        score = VERDICTS.get(verdict, 2)
        G.add_node(eid, layer="claim", verdict=verdict, verdict_score=score,
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


def edge_key(u, v):
    def short(n):
        for prefix in ("family:", "mechanism:", "claim:"):
            if n.startswith(prefix):
                return n[len(prefix):]
        return n
    a, b = short(u), short(v)
    return f"{a}--{b}" if a < b else f"{b}--{a}"


def run_node_null(G, n_perms=200):
    """Node-level curvature z-scores against degree-preserving null."""
    print(f"\n[{datetime.now():%H:%M:%S}] Node-level null model ({n_perms} permutations)")
    obs_curvatures = compute_curvatures(G)

    node_curv = {}
    for (u, v), c in obs_curvatures.items():
        node_curv.setdefault(u, []).append(c)
        node_curv.setdefault(v, []).append(c)
    obs_node_curv = {n: np.mean(cs) for n, cs in node_curv.items()}

    null_node_curvs = {n: [] for n in obs_node_curv}
    for _ in tqdm(range(n_perms), desc="Node null perms"):
        try:
            H = degree_preserving_rewire(G)
            perm_curv = compute_curvatures(H)
            nc = {}
            for (u, v), c in perm_curv.items():
                nc.setdefault(u, []).append(c)
                nc.setdefault(v, []).append(c)
            for n in null_node_curvs:
                if n in nc:
                    null_node_curvs[n].append(np.mean(nc[n]))
        except nx.NetworkXError:
            continue

    results = {}
    for n in sorted(obs_node_curv.keys()):
        obs = obs_node_curv[n]
        null = np.array(null_node_curvs[n])
        if len(null) < 10:
            continue
        z = (obs - np.mean(null)) / np.std(null) if np.std(null) > 0 else 0.0
        results[n] = {
            "obs_curvature": float(obs),
            "null_mean": float(np.mean(null)),
            "null_std": float(np.std(null)),
            "z": float(z),
            "degree": G.degree(n),
            "layer": G.nodes[n].get("layer", ""),
        }
    return results, obs_curvatures


def run_edge_null(G, obs_curvatures, n_perms=200):
    """Edge-level curvature z-scores against degree-preserving null."""
    print(f"\n[{datetime.now():%H:%M:%S}] Edge-level null model ({n_perms} permutations)")
    edges = list(G.edges())
    null_curvatures = {e: [] for e in edges}

    for _ in tqdm(range(n_perms), desc="Edge null perms"):
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
        z = (obs - np.mean(null)) / np.std(null) if np.std(null) > 0 else 0.0
        results[edge_key(u, v)] = {
            "obs": float(obs),
            "null_mean": float(np.mean(null)),
            "null_std": float(np.std(null)),
            "z": float(z),
        }
    return results


def degree_curvature_correlation(node_results):
    """Test node-level curvature vs degree correlation (confound check)."""
    print(f"\n[{datetime.now():%H:%M:%S}] Degree-curvature correlation")
    mechanism_nodes = {n: d for n, d in node_results.items()
                       if d["layer"] == "mechanism"}

    degrees = [d["degree"] for d in mechanism_nodes.values()]
    curvs = [d["obs_curvature"] for d in mechanism_nodes.values()]

    r, p = stats.pearsonr(degrees, curvs)
    rho, p_rho = stats.spearmanr(degrees, curvs)
    print(f"  Mechanism nodes (n={len(mechanism_nodes)}):")
    print(f"    Pearson  r = {r:.3f}, p = {p:.4f}")
    print(f"    Spearman ρ = {rho:.3f}, p = {p_rho:.4f}")

    return {
        "n_mechanism_nodes": len(mechanism_nodes),
        "pearson_r": float(r), "pearson_p": float(p),
        "spearman_rho": float(rho), "spearman_p": float(p_rho),
    }


def verdict_curvature_analysis(G, obs_curvatures, entries):
    """Test verdict-curvature correlation at the claim level."""
    print(f"\n[{datetime.now():%H:%M:%S}] Verdict-curvature analysis (claim-level)")

    claim_curvatures = {}
    for entry in entries:
        eid = f"claim:{entry['id']}"
        if eid not in G:
            continue
        edges_of_claim = list(G.edges(eid))
        if not edges_of_claim:
            continue
        curvs = []
        for u, v in edges_of_claim:
            c = obs_curvatures.get((u, v), obs_curvatures.get((v, u), None))
            if c is not None:
                curvs.append(c)
        if curvs:
            verdict = entry["verdict"]
            score = VERDICTS.get(verdict, 2)
            claim_curvatures[entry["id"]] = {
                "mean_curvature": float(np.mean(curvs)),
                "verdict": verdict,
                "verdict_score": score,
                "n_edges": len(curvs),
            }

    scores = [d["verdict_score"] for d in claim_curvatures.values()]
    curvs = [d["mean_curvature"] for d in claim_curvatures.values()]

    r, p = stats.pearsonr(scores, curvs)
    rho, p_rho = stats.spearmanr(scores, curvs)
    print(f"  n = {len(claim_curvatures)} claims with edges")
    print(f"  Pearson  r = {r:.3f}, p = {p:.4f}")
    print(f"  Spearman ρ = {rho:.3f}, p = {p_rho:.4f}")

    tier_stats = {}
    for score_val in sorted(set(scores)):
        tier_curvs = [c for c, s in zip(curvs, scores) if s == score_val]
        verdict_name = [k for k, v in VERDICTS.items() if v == score_val][0]
        tier_stats[verdict_name] = {
            "n": len(tier_curvs),
            "mean": float(np.mean(tier_curvs)),
            "std": float(np.std(tier_curvs)),
            "median": float(np.median(tier_curvs)),
        }
        print(f"    {verdict_name:30s}: n={len(tier_curvs):2d}, "
              f"mean={np.mean(tier_curvs):+.4f}, med={np.median(tier_curvs):+.4f}")

    jt_stat, jt_p = stats.kendalltau(scores, curvs)
    print(f"  Kendall tau = {jt_stat:.3f}, p = {jt_p:.4f}")

    return {
        "n_claims": len(claim_curvatures),
        "pearson_r": float(r), "pearson_p": float(p),
        "spearman_rho": float(rho), "spearman_p": float(p_rho),
        "kendall_tau": float(jt_stat), "kendall_p": float(jt_p),
        "tier_stats": tier_stats,
        "per_claim": claim_curvatures,
    }


def catalog_perturbation(n_bootstrap=100, subsample_frac=0.85, n_null_perms=20):
    """Subsample entries, rebuild graph, recompute z-scores."""
    subsample_size = int(len(ENTRIES) * subsample_frac)
    print(f"\n[{datetime.now():%H:%M:%S}] Catalog perturbation "
          f"({n_bootstrap} iter, {subsample_size}/{len(ENTRIES)} entries, {n_null_perms} null perms)")

    edge_survival = {}
    for i in tqdm(range(n_bootstrap), desc="Perturbation"):
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
            key = edge_key(u, v)
            if key not in edge_survival:
                edge_survival[key] = {"present": 0, "significant": 0, "z_values": []}
            edge_survival[key]["present"] += 1
            edge_survival[key]["z_values"].append(z)
            if abs(z) > 1.96:
                edge_survival[key]["significant"] += 1

    results = {}
    print(f"\n  Top surviving edges (>50% survival rate):")
    for key in sorted(edge_survival.keys()):
        s = edge_survival[key]
        if s["present"] < 10:
            continue
        rate = s["significant"] / s["present"]
        mean_z = np.mean(s["z_values"])
        if rate > 0.5:
            print(f"    {key:55s}: {s['significant']:3d}/{s['present']:3d} = {rate:.1%}  mean_z={mean_z:+.2f}")
        results[key] = {
            "n_present": s["present"],
            "n_significant": s["significant"],
            "survival_rate": float(rate),
            "mean_z": float(mean_z),
        }
    return results


def leave_one_family_out(n_null_perms=100):
    """Drop each family, recompute curvature z-scores."""
    print(f"\n[{datetime.now():%H:%M:%S}] Leave-one-family-out ({len(FAMILIES)} families)")

    G_full = build_graph()
    full_curvatures = compute_curvatures(G_full)
    full_edge_null = run_edge_null(G_full, full_curvatures, n_perms=n_null_perms)

    full_sig_edges = {k for k, v in full_edge_null.items() if abs(v["z"]) > 1.96}
    print(f"  Full graph: {len(full_sig_edges)} significant edges")

    results = {}
    for drop_family in tqdm(FAMILIES, desc="LOFO"):
        subset = [e for e in ENTRIES if e["family"] != drop_family]
        if not subset:
            continue
        G_sub = build_graph(subset)
        if G_sub.number_of_edges() < 5:
            results[drop_family] = {"n_entries_dropped": len(ENTRIES) - len(subset),
                                     "too_few_edges": True}
            continue

        sub_curvatures = compute_curvatures(G_sub)
        sub_edge_null = run_edge_null(G_sub, sub_curvatures, n_perms=n_null_perms)
        sub_sig_edges = {k for k, v in sub_edge_null.items() if abs(v["z"]) > 1.96}

        surviving = full_sig_edges & sub_sig_edges
        lost = full_sig_edges - sub_sig_edges

        results[drop_family] = {
            "n_entries_dropped": len(ENTRIES) - len(subset),
            "n_nodes": G_sub.number_of_nodes(),
            "n_edges": G_sub.number_of_edges(),
            "n_full_sig": len(full_sig_edges),
            "n_surviving_sig": len(surviving),
            "n_lost": len(lost),
            "survival_rate": len(surviving) / len(full_sig_edges) if full_sig_edges else 0,
            "lost_edges": sorted(lost),
        }
        print(f"    Drop {drop_family:25s}: {len(surviving)}/{len(full_sig_edges)} sig edges survive "
              f"({results[drop_family]['survival_rate']:.0%}), lost {len(lost)}")
    return results


def betweenness_comparison(G, obs_curvatures):
    """Compare edge curvature with edge betweenness centrality."""
    print(f"\n[{datetime.now():%H:%M:%S}] Edge betweenness vs curvature")
    edge_betweenness = nx.edge_betweenness_centrality(G)

    curvs, betws, labels = [], [], []
    for (u, v) in G.edges():
        c = obs_curvatures.get((u, v), obs_curvatures.get((v, u), None))
        b = edge_betweenness.get((u, v), edge_betweenness.get((v, u), None))
        if c is not None and b is not None:
            curvs.append(c)
            betws.append(b)
            labels.append(edge_key(u, v))

    curvs = np.array(curvs)
    betws = np.array(betws)

    r, p = stats.pearsonr(curvs, betws)
    rho, p_rho = stats.spearmanr(curvs, betws)
    print(f"  Pearson  r = {r:.3f}, p = {p:.4f}")
    print(f"  Spearman ρ = {rho:.3f}, p = {p_rho:.4f}")

    return {
        "pearson_r": float(r), "pearson_p": float(p),
        "spearman_rho": float(rho), "spearman_p": float(p_rho),
        "n_edges": len(curvs),
    }


def generate_figures(G, obs_curvatures, node_results, edge_results,
                     verdict_results, betweenness_result, fig_dir):
    """Generate all publication figures."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Degree vs curvature (mechanism nodes)
    mech = {n: d for n, d in node_results.items() if d["layer"] == "mechanism"}
    degrees = [d["degree"] for d in mech.values()]
    curvs = [d["obs_curvature"] for d in mech.values()]
    zs = [d["z"] for d in mech.values()]
    names = [n.replace("mechanism:", "") for n in mech.keys()]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["green" if abs(z) > 1.96 else "steelblue" for z in zs]
    ax.scatter(degrees, curvs, c=colors, s=60, alpha=0.7, edgecolors="black", linewidth=0.5)
    for i, name in enumerate(names):
        ax.annotate(name, (degrees[i], curvs[i]), fontsize=7, alpha=0.8,
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Node degree", fontsize=12)
    ax.set_ylabel("Mean Ollivier-Ricci curvature", fontsize=12)
    r_val = degree_curvature_correlation(node_results)
    ax.set_title(f"Expanded catalog: degree vs curvature (r={r_val['pearson_r']:.2f})", fontsize=13)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_expanded_degree_curvature.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Verdict-curvature boxplot
    if verdict_results and "per_claim" in verdict_results:
        tier_data = {}
        for cid, cd in verdict_results["per_claim"].items():
            v = cd["verdict"]
            base = v.split("/")[0].split("(")[0].strip()
            score = cd["verdict_score"]
            tier_data.setdefault(score, {"name": base, "curvs": []})
            tier_data[score]["curvs"].append(cd["mean_curvature"])

        fig, ax = plt.subplots(figsize=(10, 6))
        positions = sorted(tier_data.keys())
        data = [tier_data[p]["curvs"] for p in positions]
        labels = [f"{tier_data[p]['name']}\n(n={len(tier_data[p]['curvs'])})" for p in positions]

        bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
        palette = {1: "#e74c3c", 2: "#f39c12", 3: "#3498db", 4: "#2ecc71", 5: "#27ae60"}
        for patch, pos in zip(bp["boxes"], positions):
            patch.set_facecolor(palette.get(pos, "#95a5a6"))
            patch.set_alpha(0.6)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Mean edge curvature per claim", fontsize=12)
        ax.set_title(f"Verdict vs curvature (n={verdict_results['n_claims']}, "
                     f"τ={verdict_results['kendall_tau']:.3f}, "
                     f"p={verdict_results['kendall_p']:.4f})", fontsize=13)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_expanded_verdict_curvature.pdf", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 3. Edge curvature distribution with significance
    if edge_results:
        obs_vals = [d["obs"] for d in edge_results.values()]
        sig_vals = [d["obs"] for d in edge_results.values() if abs(d["z"]) > 1.96]
        nonsig_vals = [d["obs"] for d in edge_results.values() if abs(d["z"]) <= 1.96]

        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.linspace(min(obs_vals) - 0.05, max(obs_vals) + 0.05, 30)
        ax.hist(nonsig_vals, bins=bins, alpha=0.5, label=f"Non-significant (n={len(nonsig_vals)})",
                color="steelblue")
        ax.hist(sig_vals, bins=bins, alpha=0.7, label=f"|z| > 1.96 (n={len(sig_vals)})",
                color="tomato")
        ax.set_xlabel("Ollivier-Ricci curvature", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Edge curvature distribution (expanded catalog)", fontsize=13)
        ax.legend()
        ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_expanded_edge_distribution.pdf", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Figures saved to {fig_dir}/")


def main():
    ts_start = datetime.now()
    print(f"[{ts_start:%H:%M:%S}] Expanded curvature analysis")
    print(f"  Catalog: {len(ENTRIES)} entries, {len(FAMILIES)} families, "
          f"{len(SHARED_NODES)} mechanism types")

    G = build_graph()
    print(f"\n  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    layer_counts = {}
    for n, d in G.nodes(data=True):
        layer_counts[d.get("layer", "?")] = layer_counts.get(d.get("layer", "?"), 0) + 1
    for layer, count in sorted(layer_counts.items()):
        print(f"    {layer}: {count}")

    results = {"metadata": {
        "n_entries": len(ENTRIES), "n_families": len(FAMILIES),
        "n_mechanisms": len(SHARED_NODES),
        "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
        "timestamp": ts_start.isoformat(),
    }}

    # 1. Node-level null model
    node_results, obs_curvatures = run_node_null(G, n_perms=200)
    results["node_null"] = {n: {k: v for k, v in d.items()} for n, d in node_results.items()}

    # 2. Edge-level null model
    edge_results = run_edge_null(G, obs_curvatures, n_perms=200)
    results["edge_null"] = edge_results
    n_sig = sum(1 for v in edge_results.values() if abs(v["z"]) > 1.96)
    print(f"  {n_sig}/{len(edge_results)} edges significant at |z| > 1.96")

    # 3. Degree-curvature correlation
    results["degree_curvature"] = degree_curvature_correlation(node_results)

    # 4. Verdict-curvature analysis
    verdict_results = verdict_curvature_analysis(G, obs_curvatures, ENTRIES)
    results["verdict_curvature"] = {k: v for k, v in verdict_results.items()
                                     if k != "per_claim"}
    results["verdict_curvature_per_claim"] = verdict_results.get("per_claim", {})

    # 5. Betweenness comparison
    betw_results = betweenness_comparison(G, obs_curvatures)
    results["betweenness"] = betw_results

    # 6. Catalog perturbation
    results["catalog_perturbation"] = catalog_perturbation(
        n_bootstrap=100, subsample_frac=0.85, n_null_perms=20)

    # 7. Leave-one-family-out (major families only for speed)
    major_families = [f for f in FAMILIES
                      if sum(1 for e in ENTRIES if e["family"] == f) >= 2]
    print(f"\n  LOFO on {len(major_families)} families with >=2 entries")
    original_families = FAMILIES
    # Temporarily override for LOFO
    results["leave_one_family_out"] = {}
    for drop_family in tqdm(major_families, desc="LOFO"):
        subset = [e for e in ENTRIES if e["family"] != drop_family]
        G_sub = build_graph(subset)
        if G_sub.number_of_edges() < 5:
            results["leave_one_family_out"][drop_family] = {"too_few_edges": True}
            continue
        sub_curv = compute_curvatures(G_sub)
        sub_edge_null = run_edge_null(G_sub, sub_curv, n_perms=50)
        sub_sig = {k for k, v in sub_edge_null.items() if abs(v["z"]) > 1.96}
        full_sig = {k for k, v in edge_results.items() if abs(v["z"]) > 1.96}
        surviving = full_sig & sub_sig
        lost = full_sig - sub_sig
        n_dropped = sum(1 for e in ENTRIES if e["family"] == drop_family)
        results["leave_one_family_out"][drop_family] = {
            "n_entries_dropped": n_dropped,
            "n_nodes": G_sub.number_of_nodes(),
            "n_edges": G_sub.number_of_edges(),
            "n_surviving_sig": len(surviving),
            "n_lost": len(lost),
            "survival_rate": len(surviving) / len(full_sig) if full_sig else 0,
        }
        print(f"    Drop {drop_family:25s}: {len(surviving)}/{len(full_sig)} survive "
              f"({results['leave_one_family_out'][drop_family]['survival_rate']:.0%})")

    # Generate figures
    fig_dir = Path("psych/experiments/figures")
    generate_figures(G, obs_curvatures, node_results, edge_results,
                     verdict_results, betw_results, fig_dir)

    # Save results
    out_dir = Path("results/psych/psych/expanded")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"expanded_analysis_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = (datetime.now() - ts_start).total_seconds()
    print(f"\n[{datetime.now():%H:%M:%S}] Done in {elapsed/60:.1f} minutes")
    print(f"  Results: {out_path}")

    # Print summary
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Significant edges: {n_sig}/{len(edge_results)}")
    print(f"  Degree-curvature r = {results['degree_curvature']['pearson_r']:.3f}")
    print(f"  Verdict-curvature τ = {results['verdict_curvature']['kendall_tau']:.3f}, "
          f"p = {results['verdict_curvature']['kendall_p']:.4f}")
    print(f"  Betweenness r = {results['betweenness']['pearson_r']:.3f}")


if __name__ == "__main__":
    main()
