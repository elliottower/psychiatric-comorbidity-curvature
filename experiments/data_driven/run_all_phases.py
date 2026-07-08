"""Full 4-phase data-driven curvature analysis.

Phase 1: Data-driven graph from LDSC genetic correlations + DisGeNET
Phase 2: Method benchmarking (ORC vs Jaccard, spectral, betweenness, clustering)
Phase 3: Verdict prediction (LOOCV classification, feature importance)
Phase 4: Cross-domain replication (cardiovascular, autoimmune, oncology)

Usage:
    uv run --no-project \
        --with numpy --with scipy --with networkx --with pot \
        --with tqdm --with matplotlib --with scikit-learn --with pandas \
        python run_all_phases.py \
            --ldsc-file data/ldsc_psychiatric_rg.tsv \
            --disgenet-file data/disgenet_gda.tsv \
            --n-perms 200 \
            --output-dir ../../results/psych/data_driven
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from graph_builders import (
    build_curated_graph,
    build_ldsc_graph,
    build_disgenet_graph,
    build_domain_graph,
    DISEASE_DOMAINS,
)
from weighted_orc import run_full_pipeline
from figures import generate_all_figures, fig_cross_domain_comparison


def timestamp():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def run_phase1(args, output_dir):
    """Phase 1: Build data-driven graphs and compute curvature."""
    print(f"\n{'='*60}")
    print(f"[{timestamp()}] PHASE 1: Data-driven graph construction")
    print(f"{'='*60}\n")

    results = {}

    # 1a. Curated graph (baseline)
    print(f"[{timestamp()}] Building curated graph (baseline)...")
    G_curated, claims, verdicts, meta = build_curated_graph(expanded=True)
    r_curated = run_full_pipeline(
        G_curated, claims, verdicts,
        alpha=0.5, n_perms=args.n_perms,
        output_dir=output_dir, graph_name="curated_expanded")
    generate_all_figures(r_curated, output_dir / "figures", "Curated (84 entries)")
    results["curated"] = r_curated

    # 1b. LDSC graph (if data available)
    if args.ldsc_file and Path(args.ldsc_file).exists():
        print(f"\n[{timestamp()}] Building LDSC genetic correlation graph...")
        G_ldsc, _, _, meta_ldsc = build_ldsc_graph(
            args.ldsc_file, min_abs_rg=0.05, p_threshold=0.05)
        print(f"  LDSC graph: {G_ldsc.number_of_nodes()} nodes, {G_ldsc.number_of_edges()} edges")

        r_ldsc = run_full_pipeline(
            G_ldsc, claim_nodes=None, verdicts=None,
            alpha=0.5, n_perms=args.n_perms,
            output_dir=output_dir, graph_name="ldsc_psychiatric")
        generate_all_figures(r_ldsc, output_dir / "figures", "LDSC genetic correlations")
        results["ldsc"] = r_ldsc
    else:
        print(f"  Skipping LDSC (no file at {args.ldsc_file})")

    # 1c. DisGeNET psychiatric graph (if data available)
    if args.disgenet_file and Path(args.disgenet_file).exists():
        print(f"\n[{timestamp()}] Building DisGeNET psychiatric graph...")
        G_dg, _, _, meta_dg = build_disgenet_graph(
            args.disgenet_file,
            disease_filter=DISEASE_DOMAINS["psychiatric"],
            min_score=0.3)
        print(f"  DisGeNET psych: {G_dg.number_of_nodes()} nodes, {G_dg.number_of_edges()} edges")

        r_dg = run_full_pipeline(
            G_dg, claim_nodes=None, verdicts=None,
            alpha=0.5, n_perms=min(args.n_perms, 50),
            output_dir=output_dir, graph_name="disgenet_psychiatric")
        generate_all_figures(r_dg, output_dir / "figures", "DisGeNET psychiatric")
        results["disgenet_psych"] = r_dg
    else:
        print(f"  Skipping DisGeNET (no file at {args.disgenet_file})")

    return results


def run_phase2(phase1_results, output_dir):
    """Phase 2: Method benchmarking."""
    print(f"\n{'='*60}")
    print(f"[{timestamp()}] PHASE 2: Method benchmarking")
    print(f"{'='*60}\n")

    comparison = {}
    for graph_name, results in phase1_results.items():
        mc = results.get("method_comparison", {})
        if mc:
            comparison[graph_name] = mc
            print(f"  {graph_name}:")
            for method, vals in sorted(mc.items(), key=lambda x: abs(x[1].get("cohens_d", 0)), reverse=True):
                print(f"    {method:25s} d={vals['cohens_d']:+.3f}  p={vals['p']:.4f}")

    out_path = output_dir / "phase2_method_comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")

    return comparison


def run_phase3(phase1_results, output_dir):
    """Phase 3: Verdict prediction evaluation."""
    print(f"\n{'='*60}")
    print(f"[{timestamp()}] PHASE 3: Verdict prediction")
    print(f"{'='*60}\n")

    predictions = {}
    for graph_name, results in phase1_results.items():
        vp = results.get("verdict_prediction", {})
        if vp and "auc" in vp:
            predictions[graph_name] = {
                "auc": vp["auc"],
                "accuracy": vp["accuracy"],
                "n_positive": vp["n_positive"],
                "n_total": vp["n_total"],
            }
            print(f"  {graph_name}: AUC={vp['auc']:.3f}, Acc={vp['accuracy']:.3f}")
            print(f"    ({vp['n_positive']}/{vp['n_total']} Disconfirmed)")

    out_path = output_dir / "phase3_verdict_prediction.json"
    with open(out_path, "w") as f:
        json.dump(predictions, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")

    return predictions


def run_phase4(args, output_dir):
    """Phase 4: Cross-domain replication."""
    print(f"\n{'='*60}")
    print(f"[{timestamp()}] PHASE 4: Cross-domain replication")
    print(f"{'='*60}\n")

    if not args.disgenet_file or not Path(args.disgenet_file).exists():
        print("  Skipping Phase 4 (requires DisGeNET data)")
        return {}

    domains = ["psychiatric", "cardiovascular", "autoimmune", "oncology"]
    domain_results = {}

    for domain in domains:
        print(f"\n[{timestamp()}] Building {domain} graph...")
        G, _, _, meta = build_domain_graph(args.disgenet_file, domain, min_score=0.3)
        print(f"  {domain}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        if G.number_of_edges() < 10:
            print(f"  Too few edges, skipping")
            continue

        n_perms_domain = min(args.n_perms, 50)
        r = run_full_pipeline(
            G, claim_nodes=None, verdicts=None,
            alpha=0.5, n_perms=n_perms_domain,
            output_dir=output_dir, graph_name=f"disgenet_{domain}")
        generate_all_figures(r, output_dir / "figures", f"DisGeNET {domain}")
        domain_results[domain] = r

    if len(domain_results) >= 2:
        fig_cross_domain_comparison(
            domain_results, output_dir / "figures" / "cross_domain_comparison.pdf")

    out_path = output_dir / "phase4_cross_domain.json"
    with open(out_path, "w") as f:
        json.dump({k: v.get("metadata", {}) for k, v in domain_results.items()},
                  f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")

    return domain_results


def main():
    parser = argparse.ArgumentParser(description="4-phase data-driven curvature analysis")
    parser.add_argument("--ldsc-file", type=str, default="data/ldsc_psychiatric_rg.tsv",
                        help="Path to LDSC genetic correlations TSV")
    parser.add_argument("--disgenet-file", type=str, default="data/disgenet_gda.tsv",
                        help="Path to DisGeNET gene-disease associations TSV")
    parser.add_argument("--n-perms", type=int, default=200,
                        help="Number of permutations for null model")
    parser.add_argument("--output-dir", type=str,
                        default="../../results/psych/data_driven",
                        help="Output directory for results")
    parser.add_argument("--phases", type=str, default="1,2,3,4",
                        help="Comma-separated list of phases to run")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    phases = [int(p) for p in args.phases.split(",")]

    print(f"[{timestamp()}] Data-driven curvature analysis")
    print(f"  Phases: {phases}")
    print(f"  LDSC file: {args.ldsc_file}")
    print(f"  DisGeNET file: {args.disgenet_file}")
    print(f"  Permutations: {args.n_perms}")
    print(f"  Output: {output_dir}")

    # Phase 1: Build graphs and compute curvature
    phase1_results = {}
    if 1 in phases:
        phase1_results = run_phase1(args, output_dir)

    # Phase 2: Method benchmarking
    if 2 in phases and phase1_results:
        run_phase2(phase1_results, output_dir)

    # Phase 3: Verdict prediction
    if 3 in phases and phase1_results:
        run_phase3(phase1_results, output_dir)

    # Phase 4: Cross-domain replication
    if 4 in phases:
        run_phase4(args, output_dir)

    print(f"\n[{timestamp()}] All phases complete. Results in {output_dir}")


if __name__ == "__main__":
    main()
