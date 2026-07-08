"""Phase 1+2+3: Run full pipeline on curated graph (baseline).

This establishes the baseline: method comparison, verdict prediction,
and curvature results on the same 84-entry expert-curated graph.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph_builders import build_curated_graph
from weighted_orc import run_full_pipeline
from figures import generate_all_figures


def main():
    output_dir = Path("../../results/psych/data_driven")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    print("Building curated graph (84 entries, expanded)...")
    G, claims, verdicts, meta = build_curated_graph(expanded=True)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  {len(claims)} claims with verdicts")

    results = run_full_pipeline(
        G, claims, verdicts,
        alpha=0.5,
        n_perms=200,
        output_dir=output_dir,
        graph_name="curated_baseline",
    )

    generate_all_figures(results, output_dir / "figures", "Curated (84 entries)")

    print("\n=== SUMMARY ===")
    dc = results.get("degree_curvature", {})
    if dc:
        print(f"Degree-curvature: r={dc['r']:.3f}, p={dc['p']:.4f}")

    en = results.get("edge_null", {})
    if en:
        print(f"Edges tested: {en['n_tested']}")
        print(f"Bottleneck: {en['n_bottleneck']}, Redundant: {en['n_redundant']}")

    mc = results.get("method_comparison", {})
    if mc:
        print("\nMethod comparison (Disconfirmed vs rest):")
        for m, v in sorted(mc.items(), key=lambda x: abs(x[1].get("cohens_d", 0)), reverse=True):
            print(f"  {m:25s} d={v['cohens_d']:+.3f}  p={v['p']:.4f}")

    vp = results.get("verdict_prediction", {})
    if vp and "auc" in vp:
        print(f"\nVerdict prediction LOOCV: AUC={vp['auc']:.3f}, Acc={vp['accuracy']:.3f}")


if __name__ == "__main__":
    main()
