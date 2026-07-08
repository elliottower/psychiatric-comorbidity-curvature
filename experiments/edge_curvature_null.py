"""Edge-level curvature null model — per-edge permutation test.

For each of the 124 edges in the mechanism network, compute z-scores
against 200 degree-preserving random rewirings to identify which
specific disorder-mechanism connections are disproportionately
bottlenecked (or isolated) beyond what degree alone predicts.

Usage:
    uv run --no-project --with numpy --with scipy --with networkx --with pot --with tqdm \
        python psych/experiments/edge_curvature_null.py
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


def degree_preserving_rewire(G, n_swaps=None):
    H = G.copy()
    if n_swaps is None:
        n_swaps = H.number_of_edges() * 10
    nx.double_edge_swap(H, nswap=n_swaps, max_tries=n_swaps * 10, seed=None)
    return H


def main():
    n_perms = 200
    print(f"[{datetime.now():%H:%M:%S}] Building graph from catalog_data...")
    G = build_graph()
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print(f"[{datetime.now():%H:%M:%S}] Computing observed edge curvatures...")
    obs_curvatures = compute_curvatures(G)

    edges = list(G.edges())
    edge_keys = [f"{u}--{v}" for u, v in edges]

    # Focus on edges involving family or mechanism nodes (skip claim-claim)
    interesting_edges = []
    for u, v in edges:
        if u.startswith("family:") or u.startswith("mechanism:") or \
           v.startswith("family:") or v.startswith("mechanism:"):
            interesting_edges.append((u, v))

    print(f"  {len(interesting_edges)} edges involving family/mechanism nodes")
    print(f"  {len(edges) - len(interesting_edges)} claim-only edges (skipped for null)")

    print(f"\n[{datetime.now():%H:%M:%S}] Running {n_perms} degree-preserving permutations...")
    null_edge_curvatures = {(u, v): [] for u, v in interesting_edges}

    for i in tqdm(range(n_perms), desc="Permutations"):
        try:
            H = degree_preserving_rewire(G)
            perm_curv = compute_curvatures(H)
            for u, v in interesting_edges:
                c = perm_curv.get((u, v), perm_curv.get((v, u), None))
                if c is not None:
                    null_edge_curvatures[(u, v)].append(c)
        except nx.NetworkXError:
            continue

    n_successful = min(len(v) for v in null_edge_curvatures.values() if v) if any(null_edge_curvatures.values()) else 0
    print(f"\n  {n_successful} permutations with edge data")

    print(f"\n{'='*100}")
    print(f"  EDGE CURVATURE Z-SCORES (family/mechanism edges)")
    print(f"{'='*100}")

    results = []
    for u, v in sorted(interesting_edges, key=lambda e: obs_curvatures.get(e, obs_curvatures.get((e[1], e[0]), 0))):
        obs = obs_curvatures.get((u, v), obs_curvatures.get((v, u), 0))
        null = np.array(null_edge_curvatures[(u, v)])
        if len(null) < 10:
            continue
        null_mean = np.mean(null)
        null_std = np.std(null)
        z = (obs - null_mean) / null_std if null_std > 0 else 0.0
        p_neg = np.mean(null <= obs)

        u_short = u.replace("family:", "F:").replace("mechanism:", "M:").replace("claim:", "C:")
        v_short = v.replace("family:", "F:").replace("mechanism:", "M:").replace("claim:", "C:")
        label = f"{u_short} -- {v_short}"

        sig = "***" if abs(z) > 3 else "** " if abs(z) > 2 else "*  " if abs(z) > 1.96 else "   "
        print(f"  {label:60s}: obs={obs:+.4f} null={null_mean:+.4f}±{null_std:.4f} z={z:+.2f} {sig}")

        results.append({
            "edge": f"{u}--{v}",
            "u": u, "v": v,
            "observed": obs,
            "null_mean": null_mean,
            "null_std": null_std,
            "z_score": z,
            "p_more_negative": float(1 - p_neg),
            "n_null": len(null),
            "u_degree": G.degree(u),
            "v_degree": G.degree(v),
        })

    # Summary: most bottlenecked and most isolated edges
    results.sort(key=lambda x: x["z_score"])
    print(f"\n{'='*100}")
    print(f"  TOP 10 MOST BOTTLENECKED (negative z) EDGES")
    print(f"{'='*100}")
    for r in results[:10]:
        print(f"  {r['edge']:60s}: z={r['z_score']:+.2f} obs={r['observed']:+.4f}")

    print(f"\n{'='*100}")
    print(f"  TOP 10 MOST ISOLATED (positive z) EDGES")
    print(f"{'='*100}")
    for r in results[-10:]:
        print(f"  {r['edge']:60s}: z={r['z_score']:+.2f} obs={r['observed']:+.4f}")

    # Save
    output = {
        "n_permutations": n_successful,
        "n_edges_tested": len(results),
        "edges": {r["edge"]: {k: v for k, v in r.items() if k != "edge"} for r in results},
        "significant_bottleneck": [r for r in results if r["z_score"] < -1.96],
        "significant_isolated": [r for r in results if r["z_score"] > 1.96],
    }

    out_dir = Path("results/psych/psych/edge_curvature_null")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"edge_curvature_null_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[{datetime.now():%H:%M:%S}] Saved: {out_path}")


if __name__ == "__main__":
    main()
