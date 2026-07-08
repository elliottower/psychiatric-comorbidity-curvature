"""Expanded MechVal Psychiatric Catalog — 84 entries.

Merges the original 47-entry psych core (catalog_data.py) with 37
cross-domain MR entries from the expanded dataset.  Provides the same
interface as catalog_data.py: ENTRIES, SHARED_NODES, FAMILIES, VERDICTS.
"""
import copy

from catalog_data import (
    ENTRIES as ORIGINAL_ENTRIES,
    SHARED_NODES as ORIGINAL_SHARED_NODES,
    FAMILIES as ORIGINAL_FAMILIES,
    VERDICTS,
    FAILURE_MODES,
)

SUBSET_2_ENTRIES = [
    {"id": "045", "family": "Sleep-Mood", "claim": "Insomnia -> MDD (genetic liability)", "verdict": "Causally Suggestive", "failure_modes": ["I3"], "shared_nodes": ["circadian"]},
    {"id": "046", "family": "Sleep-Mood", "claim": "Short sleep duration -> depression", "verdict": "Causally Suggestive", "failure_modes": ["I5"], "shared_nodes": ["circadian"]},
    {"id": "047", "family": "Sleep-Mood", "claim": "Daytime napping -> MDD", "verdict": "Underdetermined", "failure_modes": ["I5"], "shared_nodes": ["circadian"]},
    {"id": "048", "family": "Sleep-Mood", "claim": "Chronotype (eveningness) -> MDD", "verdict": "Causally Suggestive", "failure_modes": ["I3"], "shared_nodes": ["circadian"]},
    {"id": "049", "family": "Eating", "claim": "Anorexia nervosa <-> metabolic/BMI", "verdict": "Mechanistically Supported", "failure_modes": ["C4"], "shared_nodes": ["metabolic"]},
    {"id": "050", "family": "Eating-Mood", "claim": "BMI -> depression (adiposity)", "verdict": "Causally Suggestive", "failure_modes": ["I5"], "shared_nodes": ["metabolic"]},
    {"id": "051", "family": "Personality", "claim": "Neuroticism -> MDD", "verdict": "Mechanistically Supported", "failure_modes": ["C4"], "shared_nodes": ["depression_convergence"]},
    {"id": "052", "family": "Personality-Social", "claim": "Loneliness -> depression", "verdict": "Causally Suggestive", "failure_modes": ["I5"], "shared_nodes": ["depression_convergence"]},
    {"id": "053", "family": "Social", "claim": "Educational attainment -> MDD (protective)", "verdict": "Causally Suggestive", "failure_modes": ["I5"], "shared_nodes": ["depression_convergence"]},
    {"id": "054", "family": "Substance-Neurodev", "claim": "Cannabis use -> schizophrenia", "verdict": "Mechanistically Supported", "failure_modes": [], "shared_nodes": ["dopamine"]},
    {"id": "055", "family": "Substance-Mood", "claim": "Smoking initiation -> depression", "verdict": "Causally Suggestive", "failure_modes": ["I5"], "shared_nodes": ["depression_convergence"]},
    {"id": "056", "family": "Substance", "claim": "Alcohol -> bipolar", "verdict": "Underdetermined", "failure_modes": ["I5"], "shared_nodes": []},
    {"id": "057", "family": "Neurodev-Mood", "claim": "ADHD liability -> MDD", "verdict": "Causally Suggestive", "failure_modes": ["C4"], "shared_nodes": ["depression_convergence"]},
    {"id": "058", "family": "Stress-Mood", "claim": "Childhood maltreatment -> MDD", "verdict": "Causally Suggestive", "failure_modes": ["I5"], "shared_nodes": ["FKBP5"]},
    {"id": "059", "family": "Inflammation", "claim": "IL-6R signaling -> depression", "verdict": "Causally Suggestive", "failure_modes": ["I7"], "shared_nodes": ["IL6_inflammation"]},
    {"id": "060", "family": "Inflammation", "claim": "CRP -> depression (BMI-mediated)", "verdict": "Disconfirmed", "failure_modes": ["I5"], "shared_nodes": ["IL6_inflammation"]},
    {"id": "061", "family": "Inflammation", "claim": "IL-6 -> schizophrenia", "verdict": "Causally Suggestive", "failure_modes": ["C4"], "shared_nodes": ["IL6_inflammation"]},
    {"id": "062", "family": "Microbiome", "claim": "Gut microbiome (taxa) -> MDD", "verdict": "Underdetermined", "failure_modes": ["C4"], "shared_nodes": ["microbiome_gut_brain"]},
    {"id": "063", "family": "Microbiome", "claim": "Microbiome -> anxiety", "verdict": "Underdetermined", "failure_modes": ["C4"], "shared_nodes": ["microbiome_gut_brain"]},
    {"id": "064", "family": "Mediator", "claim": "IL-6 -> CRP (mediation node)", "verdict": "Mechanistically Supported", "failure_modes": [], "shared_nodes": ["IL6_inflammation"]},
    {"id": "065", "family": "Microbiome-Neuro", "claim": "Gut-brain -> Parkinson's (alpha-syn)", "verdict": "Causally Suggestive", "failure_modes": ["I7"], "shared_nodes": ["microbiome_gut_brain"]},
    {"id": "066", "family": "Thyroid-Metabolic", "claim": "MDD -> type 2 diabetes (BMI-mediated)", "verdict": "Causally Suggestive", "failure_modes": ["I5"], "shared_nodes": ["metabolic"]},
    {"id": "067", "family": "Thyroid-Metabolic", "claim": "T2D -> MDD (reverse arm)", "verdict": "Underdetermined", "failure_modes": [], "shared_nodes": ["metabolic"]},
    {"id": "068", "family": "Thyroid-Metabolic", "claim": "T2D -> depression via LDL/TG/insulin", "verdict": "Causally Suggestive", "failure_modes": [], "shared_nodes": ["metabolic"]},
    {"id": "069", "family": "Thyroid-Metabolic", "claim": "Autoimmune hypothyroidism -> MDD", "verdict": "Causally Suggestive", "failure_modes": ["C4"], "shared_nodes": ["thyroid"]},
    {"id": "070", "family": "Thyroid-Metabolic", "claim": "Depression -> hypothyroidism", "verdict": "Causally Suggestive", "failure_modes": ["I5"], "shared_nodes": ["thyroid"]},
    {"id": "071", "family": "Thyroid-Metabolic", "claim": "Free thyroxine -> bipolar (protective)", "verdict": "Causally Suggestive", "failure_modes": [], "shared_nodes": ["thyroid"]},
    {"id": "072", "family": "Thyroid-Metabolic", "claim": "TSH/fT4 -> MDD (NULL control)", "verdict": "Disconfirmed", "failure_modes": [], "shared_nodes": ["thyroid"]},
    {"id": "073", "family": "Thyroid-Metabolic", "claim": "Hypothyroidism -> T2D", "verdict": "Causally Suggestive", "failure_modes": [], "shared_nodes": ["metabolic"]},
    {"id": "074", "family": "Autoimmune", "claim": "SLE -> major depression", "verdict": "Causally Suggestive", "failure_modes": ["C4"], "shared_nodes": ["autoimmune"]},
    {"id": "075", "family": "Autoimmune", "claim": "Depression -> Sjogren's", "verdict": "Causally Suggestive", "failure_modes": ["I5"], "shared_nodes": ["autoimmune"]},
    {"id": "076", "family": "Autoimmune", "claim": "Depression -> fibromyalgia", "verdict": "Mechanistically Supported", "failure_modes": [], "shared_nodes": ["autoimmune"]},
    {"id": "077", "family": "Autoimmune", "claim": "Depression -> psoriasis/PsA", "verdict": "Causally Suggestive", "failure_modes": [], "shared_nodes": ["autoimmune"]},
    {"id": "078", "family": "Autoimmune", "claim": "Bipolar -> RA (protective)", "verdict": "Underdetermined", "failure_modes": [], "shared_nodes": ["autoimmune"]},
    {"id": "079", "family": "Neuroimaging-endo", "claim": "SLE -> brain rsfMRI phenotypes", "verdict": "Causally Suggestive", "failure_modes": ["M1"], "shared_nodes": ["neuroimaging"]},
    {"id": "080", "family": "Neuroimaging-endo", "claim": "Sjogren's -> brain functional networks", "verdict": "Causally Suggestive", "failure_modes": ["M1"], "shared_nodes": ["neuroimaging"]},
    {"id": "081", "family": "Neuroimaging-endo", "claim": "Hashimoto's -> CSF metabolites/rsfMRI", "verdict": "Causally Suggestive", "failure_modes": ["M1"], "shared_nodes": ["neuroimaging"]},
]

ENTRIES = ORIGINAL_ENTRIES + SUBSET_2_ENTRIES

NEW_FAMILIES = [
    "Sleep-Mood", "Eating", "Eating-Mood", "Personality", "Personality-Social",
    "Social", "Substance-Neurodev", "Substance-Mood", "Substance", "Neurodev-Mood",
    "Stress-Mood", "Inflammation", "Microbiome", "Mediator", "Microbiome-Neuro",
    "Thyroid-Metabolic", "Autoimmune", "Neuroimaging-endo",
]

FAMILIES = ORIGINAL_FAMILIES + NEW_FAMILIES

SHARED_NODES = copy.deepcopy(ORIGINAL_SHARED_NODES)

SHARED_NODES["circadian"]["disorders"] += ["Sleep-Mood"]
SHARED_NODES["circadian"]["entries"] += ["045", "046", "047", "048"]

SHARED_NODES["dopamine"]["disorders"] += ["Substance-Neurodev"]
SHARED_NODES["dopamine"]["entries"] += ["054"]

SHARED_NODES["FKBP5"]["disorders"] += ["Stress-Mood"]
SHARED_NODES["FKBP5"]["entries"] += ["058"]

SHARED_NODES["IL6_inflammation"]["disorders"] += ["Inflammation", "Mediator"]
SHARED_NODES["IL6_inflammation"]["entries"] += ["059", "060", "061", "064"]

SHARED_NODES["metabolic"] = {
    "disorders": ["Depression", "Bipolar", "Eating", "Eating-Mood", "Thyroid-Metabolic"],
    "entries": ["013", "030", "049", "050", "066", "067", "068", "073"],
}

SHARED_NODES["depression_convergence"] = {
    "disorders": ["Depression", "Personality", "Personality-Social", "Social", "Substance-Mood", "Neurodev-Mood"],
    "entries": ["051", "052", "053", "055", "057"],
}

SHARED_NODES["microbiome_gut_brain"] = {
    "disorders": ["Microbiome", "Microbiome-Neuro"],
    "entries": ["062", "063", "065"],
}

SHARED_NODES["thyroid"] = {
    "disorders": ["Thyroid-Metabolic"],
    "entries": ["069", "070", "071", "072"],
}

SHARED_NODES["autoimmune"] = {
    "disorders": ["Autoimmune"],
    "entries": ["074", "075", "076", "077", "078"],
}

SHARED_NODES["neuroimaging"] = {
    "disorders": ["Neuroimaging-endo"],
    "entries": ["079", "080", "081"],
}
