# Psychiatric Comorbidity Curvature

Code and data for the paper:

**Edge-Level Discrete Curvature Identifies Structurally Anomalous Connections in a Psychiatric Mechanism Network**

Elliot Tower

## Summary

We apply Ollivier-Ricci curvature (ORC) to disease comorbidity and genetic correlation graphs, testing whether edge-level curvature separates within-cluster from cross-cluster connections. Across seven independent graphs spanning five data types and three disease domains, within-cluster edges consistently have higher curvature than cross-cluster edges (4/6 data-driven significant, effect sizes d = 0.46-1.55).

On an expert-curated psychiatric knowledge graph (130 nodes, 218 edges), ORC is the only edge-level metric (out of five tested) that discriminates epistemically disconfirmed claims from higher-verdict claims (d = 0.73, p = 0.011).

## Repository structure

```
paper/                       LaTeX source, references, compiled PDF
experiments/
  catalog_data.py            Expert-curated psychiatric knowledge graph
  catalog_data_expanded.py   Expanded catalog with cross-domain MR entries
  curvature_core.py          ORC computation and null model helpers
  data/
    mcgrath2020/             WHO World Mental Health Survey hazard ratios
    boschloo2015/            Boschloo et al. comorbidity data
  data_driven/
    weighted_orc.py          Weighted Ollivier-Ricci curvature (POT solver)
    run_phase1_ldsc.py       Psychiatric LDSC genetic correlations
    run_gene_sharing.py      Shared GWAS loci graph
    run_comorbidity.py       Epidemiological odds ratio graph
    run_mcgrath_comorbidity.py  WHO WMH hazard ratio graph
    run_cross_domain.py      Autoimmune + cardiometabolic LDSC
    run_unified_figure.py    Generate the 6-panel replication figure
  results/                   Pre-computed JSON results
```

## Requirements

- Python 3.9+
- networkx, numpy, scipy, matplotlib, POT (Python Optimal Transport), tqdm

Install:
```bash
pip install networkx numpy scipy matplotlib pot tqdm
```

## Running

Each analysis script is standalone:
```bash
cd experiments/data_driven
python run_phase1_ldsc.py
python run_gene_sharing.py
python run_comorbidity.py
python run_mcgrath_comorbidity.py
python run_cross_domain.py
python run_unified_figure.py   # generates the 6-panel figure
```

Results are written as timestamped JSON to `experiments/results/data_driven/`.

## Data sources

| Graph | Source | Data type |
|-------|--------|-----------|
| Psychiatric LDSC | Lee et al. 2019, Nature Genetics | Genetic correlation |
| Shared loci | Cross-Disorder Group, 2019 | GWAS overlap count |
| Comorbidity OR | Plana-Ripoll et al. 2019, JAMA Psychiatry | Odds ratio |
| WHO WMH HR | McGrath et al. 2020, JAMA Psychiatry | Hazard ratio |
| Autoimmune LDSC | Various LDSC studies | Genetic correlation |
| Cardiometabolic LDSC | Various LDSC studies | Genetic correlation |
| Expert-curated | This paper | Expert adjudication |

## License

MIT License. See [LICENSE](LICENSE).
