"""Curvature null model — permutation test for mechanism network.

Reconstructs the bipartite mechanism network from catalog_data,
computes Ollivier-Ricci curvature on 1000 degree-preserving random
graphs, and reports z-scores for each mechanism's curvature.

Also tests: does curvature add information beyond degree? (partial
correlation of curvature with verdict, controlling for degree.)

Usage:
    uv run python psych/experiments/curvature_null_model.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import networkx as nx
import ot
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from catalog_data import ENTRIES, SHARED_NODES, FAMILIES, VERDICTS


def build_graph():
    G = nx.Graph()
    for fam in FAMILIES:
        G.add_node(f"family:{fam}", layer="family")

    for node_name, info in SHARED_NODES.items():
        G.add_node(f"mechanism:{node_name}", layer="mechanism",
                   n_disorders=len(info["disorders"]))
        for disorder in info["disorders"]:
            G.add_edge(f"family:{disorder}", f"mechanism:{node_name}")

    for entry in ENTRIES:
        eid = f"claim:{entry['id']}"
        G.add_node(eid, layer="claim",
                   verdict=entry["verdict"],
                   verdict_score=VERDICTS.get(entry["verdict"], 2),
                   family=entry["family"])
        G.add_edge(eid, f"family:{entry['family']}")
        for node in entry["shared_nodes"]:
            if f"mechanism:{node}" in G:
                G.add_edge(eid, f"mechanism:{node}")
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


def node_mean_curvature(G, curvatures):
    result = {}
    for node in G.nodes():
        incident = []
        for u, v in G.edges():
            if u == node or v == node:
                incident.append(curvatures.get((u, v), curvatures.get((v, u), 0)))
        result[node] = float(np.mean(incident)) if incident else 0.0
    return result


def degree_preserving_rewire(G, n_swaps=None):
    """Degree-preserving edge rewiring (configuration model style)."""
    H = G.copy()
    if n_swaps is None:
        n_swaps = H.number_of_edges() * 10
    nx.double_edge_swap(H, nswap=n_swaps, max_tries=n_swaps * 10, seed=None)
    return H


def main():
    n_perms = 1000
    print(f"[{datetime.now():%H:%M:%S}] Building graph from catalog_data...")
    G = build_graph()
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print(f"[{datetime.now():%H:%M:%S}] Computing observed curvatures...")
    obs_curvatures = compute_curvatures(G)
    obs_node_curv = node_mean_curvature(G, obs_curvatures)

    mech_nodes = [n for n in G.nodes() if n.startswith("mechanism:")]
    family_nodes = [n for n in G.nodes() if n.startswith("family:")]
    claim_nodes = [n for n in G.nodes() if n.startswith("claim:")]

    print(f"\n  Observed mechanism curvatures:")
    for n in sorted(mech_nodes, key=lambda x: obs_node_curv[x]):
        print(f"    {n:40s}: {obs_node_curv[n]:+.4f} (degree={G.degree(n)})")

    print(f"\n[{datetime.now():%H:%M:%S}] Running {n_perms} degree-preserving permutations...")
    null_curvatures = {n: [] for n in G.nodes()}

    for i in tqdm(range(n_perms), desc="Permutations"):
        try:
            H = degree_preserving_rewire(G)
            perm_curv = compute_curvatures(H)
            perm_node_curv = node_mean_curvature(H, perm_curv)
            for n in G.nodes():
                null_curvatures[n].append(perm_node_curv.get(n, 0.0))
        except nx.NetworkXError:
            continue

    n_successful = len(null_curvatures[mech_nodes[0]])
    print(f"\n  {n_successful}/{n_perms} permutations succeeded")

    print(f"\n{'='*80}")
    print(f"  MECHANISM CURVATURE Z-SCORES (vs {n_successful} degree-preserving nulls)")
    print(f"{'='*80}")
    results = {}
    for n in sorted(mech_nodes, key=lambda x: obs_node_curv[x]):
        obs = obs_node_curv[n]
        null = np.array(null_curvatures[n])
        null_mean = np.mean(null)
        null_std = np.std(null)
        z = (obs - null_mean) / null_std if null_std > 0 else 0.0
        p_less = np.mean(null <= obs)
        name = n.replace("mechanism:", "")
        deg = G.degree(n)
        print(f"    {name:25s}: obs={obs:+.4f}  null={null_mean:+.4f}+-{null_std:.4f}  z={z:+.2f}  p(more negative)={1-p_less:.4f}  degree={deg}")
        results[name] = {
            "observed": obs, "null_mean": null_mean, "null_std": null_std,
            "z_score": z, "p_more_negative": float(1 - p_less), "degree": deg,
        }

    print(f"\n{'='*80}")
    print(f"  FAMILY NODE Z-SCORES")
    print(f"{'='*80}")
    family_results = {}
    for n in sorted(family_nodes, key=lambda x: obs_node_curv[x]):
        obs = obs_node_curv[n]
        null = np.array(null_curvatures[n])
        null_mean = np.mean(null)
        null_std = np.std(null)
        z = (obs - null_mean) / null_std if null_std > 0 else 0.0
        p_less = np.mean(null <= obs)
        name = n.replace("family:", "")
        deg = G.degree(n)
        print(f"    {name:25s}: obs={obs:+.4f}  null={null_mean:+.4f}+-{null_std:.4f}  z={z:+.2f}  p(more negative)={1-p_less:.4f}  degree={deg}")
        family_results[name] = {
            "observed": obs, "null_mean": null_mean, "null_std": null_std,
            "z_score": z, "p_more_negative": float(1 - p_less), "degree": deg,
        }

    # Degree-curvature correlation
    print(f"\n{'='*80}")
    print(f"  DEGREE-CURVATURE ANALYSIS")
    print(f"{'='*80}")
    all_nodes_data = []
    for n in G.nodes():
        all_nodes_data.append({
            "node": n,
            "degree": G.degree(n),
            "curvature": obs_node_curv[n],
            "layer": G.nodes[n].get("layer", "unknown"),
        })

    degrees = np.array([d["degree"] for d in all_nodes_data])
    curvs = np.array([d["curvature"] for d in all_nodes_data])
    r_dc, p_dc = stats.pearsonr(degrees, curvs)
    print(f"  Degree-curvature correlation (all nodes): r={r_dc:.3f}, p={p_dc:.4f}")

    mech_degrees = np.array([G.degree(n) for n in mech_nodes])
    mech_curvs = np.array([obs_node_curv[n] for n in mech_nodes])
    if len(mech_nodes) > 3:
        r_mc, p_mc = stats.pearsonr(mech_degrees, mech_curvs)
        print(f"  Degree-curvature correlation (mechanisms only): r={r_mc:.3f}, p={p_mc:.4f}")
    else:
        r_mc, p_mc = 0, 1

    # Verdict-curvature partial correlation controlling for degree
    print(f"\n{'='*80}")
    print(f"  VERDICT-CURVATURE PARTIAL CORRELATION (controlling for degree)")
    print(f"{'='*80}")
    claim_data = []
    for n in claim_nodes:
        vs = G.nodes[n].get("verdict_score", None)
        if vs is not None:
            claim_data.append({
                "node": n,
                "verdict_score": vs,
                "curvature": obs_node_curv[n],
                "degree": G.degree(n),
            })

    if len(claim_data) > 5:
        v_scores = np.array([d["verdict_score"] for d in claim_data])
        v_curvs = np.array([d["curvature"] for d in claim_data])
        v_degrees = np.array([d["degree"] for d in claim_data])

        r_vc, p_vc = stats.pearsonr(v_scores, v_curvs)
        print(f"  Raw verdict-curvature: r={r_vc:.3f}, p={p_vc:.4f}")

        r_vd, _ = stats.pearsonr(v_scores, v_degrees)
        r_cd, _ = stats.pearsonr(v_curvs, v_degrees)
        r_partial = (r_vc - r_vd * r_cd) / np.sqrt((1 - r_vd**2) * (1 - r_cd**2)) if (1 - r_vd**2) > 0 and (1 - r_cd**2) > 0 else 0
        n_obs = len(claim_data)
        t_stat = r_partial * np.sqrt((n_obs - 3) / (1 - r_partial**2)) if abs(r_partial) < 1 else 0
        p_partial = 2 * stats.t.sf(abs(t_stat), df=n_obs - 3)
        print(f"  Partial (controlling degree): r_partial={r_partial:.3f}, t={t_stat:.2f}, p={p_partial:.4f}")
        print(f"  Verdict-degree: r={r_vd:.3f}")
        print(f"  Curvature-degree: r={r_cd:.3f}")

        # By verdict tier
        print(f"\n  Per-verdict mean curvature (claims only):")
        for verdict_name, score in sorted(VERDICTS.items(), key=lambda x: x[1]):
            tier_curvs = [d["curvature"] for d in claim_data if d["verdict_score"] == score]
            if tier_curvs:
                print(f"    {verdict_name:30s} (score={score}): mean={np.mean(tier_curvs):+.4f} std={np.std(tier_curvs):.4f} n={len(tier_curvs)}")
    else:
        r_vc, p_vc, r_partial, p_partial = 0, 1, 0, 1

    # Edge curvature analysis
    print(f"\n{'='*80}")
    print(f"  EDGE CURVATURE Z-SCORES (top 10 most negative)")
    print(f"{'='*80}")
    edge_results = {}
    for (u, v), obs_c in sorted(obs_curvatures.items(), key=lambda x: x[1])[:10]:
        edge_key = f"{u}--{v}"
        null_edge = []
        for i in range(n_successful):
            try:
                H = degree_preserving_rewire(G)
                pc = compute_curvatures(H)
                if (u, v) in pc:
                    null_edge.append(pc[(u, v)])
                elif (v, u) in pc:
                    null_edge.append(pc[(v, u)])
            except Exception:
                continue
            if len(null_edge) >= 100:
                break
        if null_edge:
            nm, ns = np.mean(null_edge), np.std(null_edge)
            z = (obs_c - nm) / ns if ns > 0 else 0
            print(f"    {edge_key:60s}: obs={obs_c:+.4f} null={nm:+.4f}+-{ns:.4f} z={z:+.2f}")

    # Save results
    output = {
        "n_permutations": n_successful,
        "mechanism_results": results,
        "family_results": family_results,
        "degree_curvature_correlation": {"r": r_dc, "p": p_dc},
        "mechanism_degree_curvature_correlation": {"r": float(r_mc), "p": float(p_mc)},
        "verdict_curvature_raw": {"r": float(r_vc), "p": float(p_vc)},
        "verdict_curvature_partial": {"r_partial": float(r_partial), "p": float(p_partial)},
    }

    out_dir = Path("results/psych/psych/curvature_null_model")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"curvature_null_model_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[{datetime.now():%H:%M:%S}] Saved: {out_path}")


if __name__ == "__main__":
    main()
