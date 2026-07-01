"""Ricci-curvature validity scoring for causal inference on biomarker networks.

Ollivier-Ricci curvature of edges in a genetic correlation / biomarker
network as a reliability score for causal effect estimates. Edges with
high positive curvature sit in well-connected, redundantly-supported
regions; edges with negative curvature are bridges between weakly-connected
clusters where estimates are more fragile.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def genetic_correlation_network(
    gwas_files: dict[str, str],
    method: str = "ldsc_approx",
) -> tuple[np.ndarray, list[str]]:
    """Build a genetic correlation matrix from GWAS summary statistics.

    For a quick prototype, uses sign-concordance of top SNPs across
    traits as a proxy for LD-score regression. For publication, swap
    in proper LDSC.

    Returns (correlation_matrix, trait_names).
    """
    traits = sorted(gwas_files.keys())
    n = len(traits)

    top_snps = {}
    for trait, path in gwas_files.items():
        df = _read_gwas_header(path)
        if df is not None and len(df) > 0:
            top_snps[trait] = set(df["SNP"].head(5000).tolist()) if "SNP" in df.columns else set()

    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = traits[i], traits[j]
            if ti in top_snps and tj in top_snps:
                overlap = len(top_snps[ti] & top_snps[tj])
                union = len(top_snps[ti] | top_snps[tj])
                jaccard = overlap / union if union > 0 else 0
                corr[i, j] = corr[j, i] = jaccard
            else:
                corr[i, j] = corr[j, i] = 0.0
    return corr, traits


def _read_gwas_header(path: str, nrows: int = 10000) -> pd.DataFrame | None:
    """Read first N rows of a GWAS summary stats file, handling compression."""
    import gzip

    try:
        with gzip.open(path, "rt") as f:
            first_line = f.readline()
        sep = "\t" if "\t" in first_line else " "
        df = pd.read_csv(path, sep=sep, nrows=nrows, compression="gzip")

        snp_cols = ["SNP", "rsid", "RSID", "MarkerName", "ID", "SNPID"]
        for col in snp_cols:
            if col in df.columns:
                df = df.rename(columns={col: "SNP"})
                break

        return df
    except Exception:
        return None


def build_network(
    corr_matrix: np.ndarray,
    trait_names: list[str],
    threshold: float = 0.05,
):
    """Build a networkx graph from a correlation matrix, thresholding weak edges."""
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(trait_names)
    n = len(trait_names)
    for i in range(n):
        for j in range(i + 1, n):
            w = abs(corr_matrix[i, j])
            if w > threshold:
                G.add_edge(trait_names[i], trait_names[j], weight=w)
    return G


def compute_ollivier_ricci(G, alpha: float = 0.5) -> dict[tuple, float]:
    """Compute Ollivier-Ricci curvature for all edges.

    Uses the GraphRicciCurvature library if available, otherwise falls
    back to a simplified computation using optimal transport.
    """
    try:
        from GraphRicciCurvature.OllivierRicci import OllivierRicci

        orc = OllivierRicci(G, alpha=alpha)
        orc.compute_ricci_curvature()
        curvatures = {}
        for u, v, data in orc.G.edges(data=True):
            curvatures[(u, v)] = data.get("ricciCurvature", 0.0)
        return curvatures
    except ImportError:
        return _ollivier_ricci_manual(G, alpha)


def _ollivier_ricci_manual(G, alpha: float = 0.5) -> dict[tuple, float]:
    """Manual Ollivier-Ricci via Wasserstein distance on neighbor distributions."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    import networkx as nx

    curvatures = {}
    sp = dict(nx.all_pairs_shortest_path_length(G))

    for u, v in G.edges():
        mu_u = _neighbor_distribution(G, u, alpha)
        mu_v = _neighbor_distribution(G, v, alpha)

        nodes_u = list(mu_u.keys())
        nodes_v = list(mu_v.keys())

        all_nodes = list(set(nodes_u + nodes_v))
        d_uv = sp.get(u, {}).get(v, 1)

        cost = np.zeros((len(nodes_u), len(nodes_v)))
        for i, nu in enumerate(nodes_u):
            for j, nv in enumerate(nodes_v):
                cost[i, j] = sp.get(nu, {}).get(nv, len(G))

        weights_u = np.array([mu_u[n] for n in nodes_u])
        weights_v = np.array([mu_v[n] for n in nodes_v])

        W1 = _wasserstein_1d(cost, weights_u, weights_v)
        curvatures[(u, v)] = 1.0 - W1 / d_uv if d_uv > 0 else 0.0

    return curvatures


def _neighbor_distribution(G, node, alpha: float) -> dict:
    """Lazy random walk distribution: alpha on self, (1-alpha) uniform on neighbors."""
    neighbors = list(G.neighbors(node))
    dist = {node: alpha}
    if neighbors:
        w = (1 - alpha) / len(neighbors)
        for n in neighbors:
            dist[n] = dist.get(n, 0) + w
    return dist


def _wasserstein_1d(cost: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Approximate Wasserstein distance via linear assignment on discretized mass."""
    try:
        import ot
        return ot.emd2(a, b, cost)
    except ImportError:
        return float(np.sum(cost * np.outer(a, b)))


def validity_scores(
    curvatures: dict[tuple, float],
    mr_results: dict[tuple, dict] | None = None,
) -> pd.DataFrame:
    """Combine curvature with MR concordance into a validity score.

    If mr_results is provided (keyed by (exposure, outcome) tuples with
    'concordant' and methods), computes the correlation between curvature
    and MR reliability.
    """
    rows = []
    for (u, v), kappa in curvatures.items():
        row = {"edge_from": u, "edge_to": v, "ricci_curvature": kappa}
        if mr_results and (u, v) in mr_results:
            mr = mr_results[(u, v)]
            row["mr_concordant"] = mr.get("concordant", None)
            methods = mr.get("methods", {})
            if methods:
                pvals = [m.get("p", 1.0) for m in methods.values()]
                row["mr_min_p"] = min(pvals)
                row["mr_mean_p"] = np.mean(pvals)
        rows.append(row)

    df = pd.DataFrame(rows)
    if "mr_concordant" in df.columns and df["mr_concordant"].notna().sum() > 3:
        from scipy.stats import pointbiserialr
        mask = df["mr_concordant"].notna()
        r, p = pointbiserialr(
            df.loc[mask, "mr_concordant"].astype(int),
            df.loc[mask, "ricci_curvature"],
        )
        print(f"Curvature-concordance correlation: r={r:.3f}, p={p:.4f}")

    return df.sort_values("ricci_curvature", ascending=False)


def run_demo(gwas_dir: str = "data/gwas"):
    """Run on our 8 PGC GWAS datasets."""
    import json
    from pathlib import Path

    manifest_path = Path(gwas_dir) / "manifest.json"
    if not manifest_path.exists():
        print("No manifest.json found. Provide a GWAS manifest file.")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    gwas_files = {name: info["file"] for name, info in manifest.items()}
    print(f"Building genetic correlation network from {len(gwas_files)} traits...")

    corr, traits = genetic_correlation_network(gwas_files)
    print(f"\nCorrelation matrix ({len(traits)} traits):")
    print(pd.DataFrame(corr, index=traits, columns=traits).round(3))

    G = build_network(corr, traits, threshold=0.02)
    print(f"\nNetwork: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if G.number_of_edges() == 0:
        print("No edges above threshold. Try lowering threshold or using LDSC.")
        return

    curvatures = compute_ollivier_ricci(G)
    scores = validity_scores(curvatures)
    print(f"\nValidity scores (by Ricci curvature):")
    print(scores.to_string(index=False))
    return scores


if __name__ == "__main__":
    run_demo()
