"""Graph builders for data-driven analysis.

Each builder function returns:
  (G, claim_nodes, verdicts, metadata)

where G is a nx.Graph, claim_nodes are nodes with verdict labels,
verdicts maps node -> verdict string, and metadata is a dict.
"""

import csv
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# 1. Expert-curated catalog (baseline — the current paper's graph)
# ---------------------------------------------------------------------------

def build_curated_graph(expanded=True):
    """Build graph from the expert-curated catalog.

    This is the same graph as the current paper. Used as a baseline
    for comparison with data-driven graphs.
    """
    if expanded:
        from catalog_data_expanded import ENTRIES, SHARED_NODES, FAMILIES, VERDICTS
    else:
        from catalog_data import ENTRIES, SHARED_NODES, FAMILIES, VERDICTS

    G = nx.Graph()

    for entry in ENTRIES:
        claim_id = f"claim_{entry['id']}"
        family = entry["family"]

        G.add_node(claim_id, node_type="claim", verdict=entry["verdict"],
                   family=family, claim=entry["claim"])
        G.add_node(family, node_type="family")
        G.add_edge(claim_id, family)

        for mech in entry.get("shared_nodes", []):
            G.add_node(mech, node_type="mechanism")
            G.add_edge(claim_id, mech)

    for mech_name, mech_data in SHARED_NODES.items():
        G.add_node(mech_name, node_type="mechanism")
        for disorder in mech_data["disorders"]:
            if G.has_node(disorder):
                G.add_edge(mech_name, disorder)

    claim_nodes = [n for n in G.nodes() if G.nodes[n].get("node_type") == "claim"]
    verdicts_map = {n: G.nodes[n]["verdict"] for n in claim_nodes}

    return G, claim_nodes, verdicts_map, {
        "source": "expert_curated",
        "expanded": expanded,
        "n_entries": len(ENTRIES),
    }


# ---------------------------------------------------------------------------
# 2. LDSC genetic correlations (disorder-disorder weighted graph)
# ---------------------------------------------------------------------------

def build_ldsc_graph(rg_file, min_abs_rg=0.0, p_threshold=1.0):
    """Build weighted disorder-disorder graph from LDSC genetic correlations.

    Args:
        rg_file: path to TSV/CSV with columns: trait1, trait2, rg, se, p
        min_abs_rg: minimum |rg| to include edge
        p_threshold: maximum p-value to include edge

    Edge weights = |rg| (absolute genetic correlation).
    Sign stored as edge attribute 'rg_sign'.
    """
    G = nx.Graph()

    with open(rg_file) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            t1 = row["trait1"].strip()
            t2 = row["trait2"].strip()
            rg = float(row["rg"])
            p = float(row.get("p", 0))

            if abs(rg) < min_abs_rg:
                continue
            if p > p_threshold:
                continue

            G.add_node(t1, node_type="trait")
            G.add_node(t2, node_type="trait")
            G.add_edge(t1, t2, weight=abs(rg), rg=rg, rg_sign=np.sign(rg), p=p)

    return G, [], {}, {
        "source": "ldsc",
        "rg_file": str(rg_file),
        "min_abs_rg": min_abs_rg,
        "p_threshold": p_threshold,
    }


# ---------------------------------------------------------------------------
# 3. DisGeNET gene-disease associations (bipartite weighted graph)
# ---------------------------------------------------------------------------

def build_disgenet_graph(gda_file, disease_filter=None, min_score=0.0,
                         collapse_genes_to_pathways=False):
    """Build weighted disease-gene graph from DisGeNET GDA file.

    Args:
        gda_file: path to DisGeNET all_gene_disease_associations.tsv
        disease_filter: optional list of disease CUI prefixes or names to include
        min_score: minimum GDA score to include
        collapse_genes_to_pathways: if True, group genes by pathway (requires
            separate pathway mapping file)

    Edge weights = GDA score (0-1, higher = more evidence).
    """
    G = nx.Graph()
    skipped = 0

    with open(gda_file) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gene = row.get("geneSymbol", row.get("gene_symbol", "")).strip()
            disease = row.get("diseaseName", row.get("disease_name", "")).strip()
            score = float(row.get("score", row.get("gda_score", 0)))
            disease_class = row.get("diseaseClass", row.get("disease_class", ""))

            if score < min_score:
                skipped += 1
                continue

            if disease_filter:
                match = any(f.lower() in disease.lower() or
                           f.lower() in disease_class.lower()
                           for f in disease_filter)
                if not match:
                    skipped += 1
                    continue

            G.add_node(gene, node_type="gene")
            G.add_node(disease, node_type="disease", disease_class=disease_class)
            G.add_edge(gene, disease, weight=score)

    return G, [], {}, {
        "source": "disgenet",
        "n_genes": sum(1 for n in G.nodes() if G.nodes[n].get("node_type") == "gene"),
        "n_diseases": sum(1 for n in G.nodes() if G.nodes[n].get("node_type") == "disease"),
        "min_score": min_score,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# 4. Open Targets disease-target associations
# ---------------------------------------------------------------------------

def build_opentargets_graph(associations_file, disease_ids=None, min_score=0.0):
    """Build weighted disease-target graph from Open Targets bulk data.

    Args:
        associations_file: path to Open Targets overall associations parquet/JSON
        disease_ids: optional list of EFO IDs to filter (e.g., psychiatric disorders)
        min_score: minimum overall association score

    Edge weights = overall association score.
    """
    G = nx.Graph()

    try:
        import pandas as pd
        df = pd.read_parquet(associations_file)
    except Exception:
        with open(associations_file) as f:
            records = [json.loads(line) for line in f]
        import pandas as pd
        df = pd.DataFrame(records)

    if disease_ids:
        df = df[df["diseaseId"].isin(disease_ids)]
    if min_score > 0:
        df = df[df["score"] >= min_score]

    for _, row in df.iterrows():
        target = row.get("targetSymbol", row.get("targetId", ""))
        disease = row.get("diseaseName", row.get("diseaseId", ""))
        score = float(row.get("score", 0))

        G.add_node(target, node_type="target")
        G.add_node(disease, node_type="disease")
        G.add_edge(target, disease, weight=score)

    return G, [], {}, {
        "source": "opentargets",
        "n_targets": sum(1 for n in G.nodes() if G.nodes[n].get("node_type") == "target"),
        "n_diseases": sum(1 for n in G.nodes() if G.nodes[n].get("node_type") == "disease"),
        "min_score": min_score,
    }


# ---------------------------------------------------------------------------
# 5. PheWAS cross-phenotype associations
# ---------------------------------------------------------------------------

def build_phewas_graph(phewas_file, p_threshold=5e-8, min_abs_beta=0.0):
    """Build weighted phenotype-SNP graph from PheWAS catalog data.

    Args:
        phewas_file: path to PheWAS catalog TSV
        p_threshold: maximum p-value for inclusion
        min_abs_beta: minimum |beta| for inclusion

    Useful for cross-phenotype analysis — SNPs shared across phenotypes
    create indirect phenotype-phenotype connections.
    """
    G = nx.Graph()

    with open(phewas_file) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            snp = row.get("snp", row.get("SNP", "")).strip()
            phenotype = row.get("phewas_phenotype", row.get("phenotype", "")).strip()
            p = float(row.get("p", row.get("p_value", 1)))
            beta = float(row.get("beta", row.get("odds_ratio", 0)))

            if p > p_threshold:
                continue
            if abs(beta) < min_abs_beta:
                continue

            G.add_node(snp, node_type="snp")
            G.add_node(phenotype, node_type="phenotype")
            G.add_edge(snp, phenotype, weight=-np.log10(max(p, 1e-300)), p=p, beta=beta)

    return G, [], {}, {
        "source": "phewas",
        "p_threshold": p_threshold,
    }


# ---------------------------------------------------------------------------
# 6. Cross-domain builder (Phase 4)
# ---------------------------------------------------------------------------

DISEASE_DOMAINS = {
    "psychiatric": [
        "depression", "bipolar", "schizophrenia", "anxiety", "ptsd",
        "ocd", "adhd", "autism", "addiction", "substance",
        "anorexia", "bulimia", "insomnia", "psychosis",
    ],
    "cardiovascular": [
        "coronary", "heart", "cardiac", "hypertension", "atherosclerosis",
        "myocardial", "stroke", "arrhythmia", "heart failure", "angina",
    ],
    "autoimmune": [
        "lupus", "rheumatoid", "multiple sclerosis", "crohn", "colitis",
        "psoriasis", "sjogren", "celiac", "type 1 diabetes", "scleroderma",
    ],
    "oncology": [
        "cancer", "carcinoma", "melanoma", "leukemia", "lymphoma",
        "tumor", "neoplasm", "glioma", "sarcoma", "myeloma",
    ],
}


def build_domain_graph(gda_file, domain, min_score=0.3):
    """Build a domain-specific graph from DisGeNET for cross-domain replication.

    Args:
        gda_file: path to DisGeNET all_gene_disease_associations.tsv
        domain: one of 'psychiatric', 'cardiovascular', 'autoimmune', 'oncology'
        min_score: minimum GDA score
    """
    keywords = DISEASE_DOMAINS.get(domain, [])
    if not keywords:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(DISEASE_DOMAINS.keys())}")

    return build_disgenet_graph(gda_file, disease_filter=keywords, min_score=min_score)
