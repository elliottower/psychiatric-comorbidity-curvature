"""ORC analysis on larger published psychiatric networks.

Analyzes:
1. Boschloo et al. (2015) — 120 DSM symptom nodes, NESARC Wave 2 (N=34,653)
2. McGrath et al. (2020) — 22 disorder nodes, WHO WMH (N=145,990)
3. Dervic et al. (2025) — ICD-10 psychiatric sub-network from Austrian hospital data

Results saved to results/larger_networks/
"""
import json
import numpy as np
import networkx as nx
import ot
from scipy import stats
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


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
    for u, v in tqdm(list(G.edges()), desc="Computing ORC", leave=False):
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
        result[v] = float(np.mean([curvatures[(a, b)] for a, b in edges]))
    return result


def classify_node(kappa, threshold=0.05):
    if kappa < -threshold:
        return "bridge"
    elif kappa > threshold:
        return "cluster-internal"
    else:
        return "mixed"


def analyze_network(G, name):
    """Full ORC analysis of a network: curvatures, hub-bridge decomposition, community detection."""
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Analyzing: {name}")
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"  Connected: {nx.is_connected(G)}")
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        print(f"  Components: {len(components)}, largest: {max(len(c) for c in components)}")
        G = G.subgraph(max(components, key=len)).copy()
        print(f"  Using largest component: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    orc = compute_orc(G, alpha=0.5)
    mc = mean_curvature(G, orc)

    nodes_sorted = sorted(mc.items(), key=lambda x: x[1])
    degrees = dict(G.degree())
    betweenness = nx.betweenness_centrality(G)
    clustering = nx.clustering(G)

    n_bridge = sum(1 for v in mc if mc[v] < -0.05)
    n_mixed = sum(1 for v in mc if -0.05 <= mc[v] <= 0.05)
    n_cluster = sum(1 for v in mc if mc[v] > 0.05)

    print(f"\n  Classification (threshold ±0.05):")
    print(f"    Bridge: {n_bridge}, Mixed: {n_mixed}, Cluster-internal: {n_cluster}")

    print(f"\n  Top 10 bridges (most negative κ̄):")
    for node, kappa in nodes_sorted[:10]:
        print(f"    {node:30s}  κ̄={kappa:+.4f}  deg={degrees[node]:3d}  "
              f"betw={betweenness[node]:.4f}  clust={clustering[node]:.3f}")

    print(f"\n  Top 10 cluster-internal (most positive κ̄):")
    for node, kappa in nodes_sorted[-10:][::-1]:
        print(f"    {node:30s}  κ̄={kappa:+.4f}  deg={degrees[node]:3d}  "
              f"betw={betweenness[node]:.4f}  clust={clustering[node]:.3f}")

    orc_vals = np.array([mc[v] for v in sorted(G.nodes())])
    deg_vals = np.array([degrees[v] for v in sorted(G.nodes())])
    btw_vals = np.array([betweenness[v] for v in sorted(G.nodes())])
    clu_vals = np.array([clustering[v] for v in sorted(G.nodes())])

    corr_deg = stats.spearmanr(orc_vals, deg_vals)
    corr_btw = stats.spearmanr(orc_vals, btw_vals)
    corr_clu = stats.spearmanr(orc_vals, clu_vals)

    forman_vals = np.array([
        np.mean([4 - degrees[u] - degrees[v] for u in G.neighbors(v)])
        for v in sorted(G.nodes())
    ])
    corr_forman_deg = stats.spearmanr(forman_vals, deg_vals)

    print(f"\n  Correlations:")
    print(f"    ORC-degree: ρ={corr_deg.correlation:.3f} (p={corr_deg.pvalue:.4f})")
    print(f"    ORC-betweenness: ρ={corr_btw.correlation:.3f} (p={corr_btw.pvalue:.4f})")
    print(f"    ORC-clustering: ρ={corr_clu.correlation:.3f} (p={corr_clu.pvalue:.4f})")
    print(f"    Forman-degree: ρ={corr_forman_deg.correlation:.3f} (p={corr_forman_deg.pvalue:.4f})")

    per_node = []
    for v in sorted(G.nodes()):
        per_node.append({
            "node": v,
            "kappa": mc[v],
            "degree": degrees[v],
            "betweenness": float(betweenness[v]),
            "clustering": float(clustering[v]),
            "classification": classify_node(mc[v]),
        })

    edge_curvatures = []
    for u, v in G.edges():
        edge_curvatures.append({
            "edge": f"{u}--{v}",
            "curvature": float(orc[(u, v)]),
        })

    return {
        "name": name,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_bridge": n_bridge,
        "n_mixed": n_mixed,
        "n_cluster_internal": n_cluster,
        "mean_curvature": float(np.mean(orc_vals)),
        "std_curvature": float(np.std(orc_vals)),
        "correlations": {
            "orc_degree": {"rho": float(corr_deg.correlation), "p": float(corr_deg.pvalue)},
            "orc_betweenness": {"rho": float(corr_btw.correlation), "p": float(corr_btw.pvalue)},
            "orc_clustering": {"rho": float(corr_clu.correlation), "p": float(corr_clu.pvalue)},
            "forman_degree": {"rho": float(corr_forman_deg.correlation), "p": float(corr_forman_deg.pvalue)},
        },
        "top10_bridges": [{"node": n, "kappa": k} for n, k in nodes_sorted[:10]],
        "top10_cluster": [{"node": n, "kappa": k} for n, k in nodes_sorted[-10:][::-1]],
        "per_node": per_node,
        "edge_curvatures": edge_curvatures,
    }


# ============================================================
# Network loaders
# ============================================================

def load_boschloo_network(data_dir="data/boschloo2015"):
    """Load Boschloo et al. (2015) 120-node Ising network from S2 XLSX.

    XLSX structure: row 0 = disorder names (merged cells), row 1 = symptom numbers,
    rows 2+ = 120x120 edge weight matrix. Cols 0-1 are row labels.
    """
    import openpyxl
    xlsx_path = Path(data_dir)
    xlsx_files = list(xlsx_path.glob("*.xlsx"))
    if not xlsx_files:
        print(f"  No XLSX found in {data_dir}, skipping Boschloo")
        return None

    wb = openpyxl.load_workbook(xlsx_files[0], data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))

    disorder_row = list(rows[0])
    number_row = list(rows[1])

    current_disorder = None
    node_names = []
    for i in range(2, len(disorder_row)):
        if disorder_row[i] is not None:
            current_disorder = str(disorder_row[i]).strip()
        num = number_row[i]
        if num is not None:
            node_names.append(f"{current_disorder}_{num}")

    G = nx.Graph()
    for i, row in enumerate(rows[2:]):
        if i >= len(node_names):
            break
        for j in range(2, len(row)):
            col_idx = j - 2
            if col_idx >= len(node_names) or col_idx <= i:
                continue
            val = row[j]
            if val is not None and val != 0 and val != "":
                try:
                    w = float(val)
                    if abs(w) > 0.001:
                        G.add_edge(node_names[i], node_names[col_idx], weight=w)
                except (ValueError, TypeError):
                    pass

    print(f"  Loaded Boschloo: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def load_mcgrath_network(data_dir="data/mcgrath2020"):
    """Load McGrath et al. (2020) WHO WMH comorbidity network from CSV."""
    import csv
    csv_path = Path(data_dir)
    csv_files = list(csv_path.glob("*.csv"))
    if not csv_files:
        print(f"  No CSV found in {data_dir}, skipping McGrath")
        return None

    G = nx.Graph()
    with open(csv_files[0], 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prior = row.get("Prior disorder", row.get("Prior", "")).strip()
            later = row.get("Later disorder", row.get("Later", "")).strip()
            hr = row.get("HR", "")
            sex = row.get("Sex", "").strip()
            model = row.get("Model", "").strip()

            if sex not in ("All", "all", "Total", "total", "Both", "both", ""):
                continue
            if model not in ("A", "a", ""):
                continue

            try:
                hr_val = float(hr)
            except (ValueError, TypeError):
                continue

            if prior and later and hr_val > 1.0:
                if G.has_edge(prior, later):
                    existing = G[prior][later].get('weight', 1.0)
                    G[prior][later]['weight'] = max(existing, hr_val)
                else:
                    G.add_edge(prior, later, weight=hr_val)

    print(f"  Loaded McGrath: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def load_dervic_network(data_dir="data/dervic2025"):
    """Load Dervic et al. (2025) ICD-10 psychiatric sub-network."""
    import csv
    data_path = Path(data_dir)
    csv_files = list(data_path.rglob("*.csv"))
    if not csv_files:
        print(f"  No CSV found in {data_dir}, skipping Dervic")
        return None

    adj_files = [f for f in csv_files if "djacenc" in f.name.lower() or "adj" in f.name.lower()]
    if not adj_files:
        adj_files = csv_files

    target = adj_files[0]
    print(f"  Reading: {target.name}")

    G_full = nx.Graph()
    with open(target, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header:
            codes = [h.strip() for h in header[1:] if h.strip()]
            for i, row in enumerate(reader):
                if i >= len(codes):
                    break
                for j in range(1, len(row)):
                    if j - 1 >= len(codes) or j - 1 <= i:
                        continue
                    try:
                        val = float(row[j])
                        if val > 0:
                            G_full.add_edge(codes[i], codes[j-1], weight=val)
                    except (ValueError, TypeError):
                        pass

    psych_nodes = [n for n in G_full.nodes() if str(n).startswith("F")]
    if not psych_nodes:
        print(f"  No F-code nodes found, trying all nodes")
        return G_full if G_full.number_of_nodes() > 0 else None

    G_psych = G_full.subgraph(psych_nodes).copy()
    print(f"  Psychiatric sub-network (F-codes): {G_psych.number_of_nodes()} nodes, {G_psych.number_of_edges()} edges")
    return G_psych


# ============================================================
# Our 23-node network (for comparison)
# ============================================================

EDGES_23 = [
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

def build_23node_graph():
    G = nx.Graph()
    for a, b in EDGES_23:
        G.add_edge(a, b)
    return G


if __name__ == "__main__":
    out_dir = Path("results/larger_networks")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting larger network ORC analysis")

    # Our 23-node network as baseline
    G23 = build_23node_graph()
    all_results["our_23node"] = analyze_network(G23, "Our 23-node comorbidity network")

    # Boschloo 120-node
    G_bosch = load_boschloo_network()
    if G_bosch is not None and G_bosch.number_of_nodes() > 0:
        all_results["boschloo_120"] = analyze_network(G_bosch, "Boschloo et al. (2015) 120-node DSM symptoms")

    # McGrath 22-node
    G_mcg = load_mcgrath_network()
    if G_mcg is not None and G_mcg.number_of_nodes() > 0:
        all_results["mcgrath_22"] = analyze_network(G_mcg, "McGrath et al. (2020) WHO WMH 22 disorders")

    # Dervic ICD-10 psychiatric
    G_derv = load_dervic_network()
    if G_derv is not None and G_derv.number_of_nodes() > 0:
        all_results["dervic_icd10_psych"] = analyze_network(G_derv, "Dervic et al. (2025) ICD-10 psychiatric sub-network")

    # Summary comparison
    print(f"\n{'='*60}")
    print(f"SUMMARY COMPARISON")
    print(f"{'='*60}")
    for key, res in all_results.items():
        print(f"\n{res['name']}:")
        print(f"  {res['n_nodes']} nodes, {res['n_edges']} edges")
        print(f"  Bridge/Mixed/Cluster: {res['n_bridge']}/{res['n_mixed']}/{res['n_cluster_internal']}")
        print(f"  Mean κ̄: {res['mean_curvature']:.4f} ± {res['std_curvature']:.4f}")
        print(f"  ORC-degree ρ: {res['correlations']['orc_degree']['rho']:.3f}")
        print(f"  Forman-degree ρ: {res['correlations']['forman_degree']['rho']:.3f}")
        print(f"  Top bridge: {res['top10_bridges'][0]['node']} (κ̄={res['top10_bridges'][0]['kappa']:.4f})")

    out_path = out_dir / "larger_network_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Saved to {out_path}")
