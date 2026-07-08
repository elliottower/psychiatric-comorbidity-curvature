"""MechVal Psychiatric Catalog — 44 entries as structured data.

Machine-readable version of the catalog for computational experiments.
Each entry: (id, family, claim, verdict, failure_modes, shared_nodes, upgrade_class).
"""

VERDICTS = {
    "Validated": 5,
    "Mechanistically Supported": 4,
    "Causally Suggestive": 3,
    "Underdetermined": 2,
    "Inconclusive": 2,
    "Disconfirmed": 1,
}

ENTRIES = [
    {"id": "001", "family": "Depression", "claim": "Treatment-Resistant Depression construct", "verdict": "Disconfirmed", "failure_modes": ["C4", "C1"], "shared_nodes": [], "upgrade_class": "U4"},
    {"id": "003", "family": "Depression", "claim": "Serotonin/chemical-imbalance", "verdict": "Disconfirmed", "failure_modes": ["F9", "I1"], "shared_nodes": ["serotonin"], "upgrade_class": None},
    {"id": "004", "family": "Depression", "claim": "HPA cortisol -> depression", "verdict": "Disconfirmed", "failure_modes": ["I1", "I5"], "shared_nodes": ["HPA_axis", "FKBP5"], "upgrade_class": "U4"},
    {"id": "006a", "family": "Depression", "claim": "IL-6 -> MDD", "verdict": "Causally Suggestive", "failure_modes": ["I7"], "shared_nodes": ["IL6_inflammation"], "upgrade_class": "U1"},
    {"id": "006b", "family": "Depression", "claim": "CRP -> MDD", "verdict": "Disconfirmed", "failure_modes": ["I5"], "shared_nodes": ["IL6_inflammation"], "upgrade_class": None},
    {"id": "008", "family": "Depression", "claim": "BDNF etiological", "verdict": "Disconfirmed", "failure_modes": ["I1", "I5", "M1"], "shared_nodes": [], "upgrade_class": None},
    {"id": "008b", "family": "Depression", "claim": "BDNF/TrkB drug-mechanism", "verdict": "Mechanistically Supported", "failure_modes": [], "shared_nodes": [], "upgrade_class": None},
    {"id": "010", "family": "Depression", "claim": "Burnout distinctness", "verdict": "Inconclusive", "failure_modes": ["C4"], "shared_nodes": [], "upgrade_class": "U4"},
    {"id": "011", "family": "Depression", "claim": "Kynurenine/QA-KynA pathway", "verdict": "Causally Suggestive", "failure_modes": ["I3"], "shared_nodes": ["NMDA_glutamate", "IL6_inflammation"], "upgrade_class": "U3"},
    {"id": "012", "family": "Depression", "claim": "Glutamate/NMDA etiology", "verdict": "Underdetermined", "failure_modes": ["I1", "I3", "C2"], "shared_nodes": ["NMDA_glutamate"], "upgrade_class": "U3"},
    {"id": "012a", "family": "Depression", "claim": "Glutamatergic treatment target", "verdict": "Mechanistically Supported", "failure_modes": [], "shared_nodes": ["NMDA_glutamate"], "upgrade_class": None},
    {"id": "013", "family": "Depression", "claim": "Mitochondrial/bioenergetic", "verdict": "Underdetermined", "failure_modes": ["I1", "I5"], "shared_nodes": ["mitochondrial"], "upgrade_class": "U3"},
    {"id": "014", "family": "Depression", "claim": "Circadian/sleep-disruption", "verdict": "Causally Suggestive", "failure_modes": ["I3"], "shared_nodes": ["circadian"], "upgrade_class": "U1"},
    {"id": "015", "family": "Depression", "claim": "GR-resistance/FKBP5 (GxE)", "verdict": "Causally Suggestive", "failure_modes": ["C3"], "shared_nodes": ["FKBP5", "HPA_axis"], "upgrade_class": "U1"},
    {"id": "016", "family": "Depression", "claim": "Folate/one-carbon (MTHFR)", "verdict": "Causally Suggestive", "failure_modes": ["M1", "I3"], "shared_nodes": [], "upgrade_class": "U1"},
    {"id": "017", "family": "PTSD", "claim": "FKBP5/GR-sensitivity (GxE)", "verdict": "Causally Suggestive", "failure_modes": ["C3"], "shared_nodes": ["FKBP5", "HPA_axis"], "upgrade_class": "U1"},
    {"id": "018", "family": "PTSD", "claim": "Fear-conditioning/amygdala-PFC circuit", "verdict": "Mechanistically Supported", "failure_modes": ["C4"], "shared_nodes": ["fear_circuit"], "upgrade_class": None},
    {"id": "019", "family": "PTSD", "claim": "Noradrenergic/LC hyperarousal", "verdict": "Causally Suggestive", "failure_modes": ["I7"], "shared_nodes": ["noradrenergic"], "upgrade_class": "U2"},
    {"id": "020", "family": "PTSD", "claim": "Low-cortisol/HPA-sensitization", "verdict": "Underdetermined", "failure_modes": ["M1", "M2"], "shared_nodes": ["HPA_axis"], "upgrade_class": "U3"},
    {"id": "021a", "family": "Anxiety", "claim": "CO2 panic phenomenon", "verdict": "Validated", "failure_modes": [], "shared_nodes": [], "upgrade_class": None},
    {"id": "021b", "family": "Anxiety", "claim": "Chemoreflex suffocation-monitor", "verdict": "Disconfirmed", "failure_modes": [], "shared_nodes": [], "upgrade_class": None},
    {"id": "022", "family": "Anxiety", "claim": "Fear-circuit/amygdala (sustained anxiety)", "verdict": "Mechanistically Supported", "failure_modes": ["C4"], "shared_nodes": ["fear_circuit"], "upgrade_class": None},
    {"id": "023", "family": "Anxiety", "claim": "Serotonergic deficit model (anxiety)", "verdict": "Disconfirmed", "failure_modes": ["F9"], "shared_nodes": ["serotonin"], "upgrade_class": None},
    {"id": "024", "family": "Anxiety", "claim": "Noradrenergic/orexin hyperarousal", "verdict": "Causally Suggestive", "failure_modes": ["I7"], "shared_nodes": ["noradrenergic"], "upgrade_class": "U2"},
    {"id": "025", "family": "Schizophrenia", "claim": "Revised dopamine (presynaptic/associative)", "verdict": "Mechanistically Supported", "failure_modes": ["C4"], "shared_nodes": ["dopamine"], "upgrade_class": None},
    {"id": "025-orig", "family": "Schizophrenia", "claim": "Global hyperdopaminergia etiology", "verdict": "Disconfirmed", "failure_modes": ["I5", "C4"], "shared_nodes": ["dopamine"], "upgrade_class": None},
    {"id": "026", "family": "Schizophrenia", "claim": "NMDA-hypofunction/glutamate", "verdict": "Causally Suggestive", "failure_modes": ["I7", "I3"], "shared_nodes": ["NMDA_glutamate"], "upgrade_class": "U2"},
    {"id": "027", "family": "Schizophrenia", "claim": "Complement C4/excessive pruning", "verdict": "Mechanistically Supported", "failure_modes": ["I7"], "shared_nodes": ["synaptic_pruning"], "upgrade_class": "U2"},
    {"id": "028", "family": "Schizophrenia", "claim": "Neurodevelopmental/two-hit", "verdict": "Causally Suggestive", "failure_modes": ["C1"], "shared_nodes": ["MIA_prenatal"], "upgrade_class": "U4"},
    {"id": "029", "family": "Schizophrenia", "claim": "Maternal immune activation/IL-6", "verdict": "Causally Suggestive", "failure_modes": ["I5", "C4"], "shared_nodes": ["MIA_prenatal", "IL6_inflammation"], "upgrade_class": "U1"},
    {"id": "030", "family": "Bipolar", "claim": "Mitochondrial/bioenergetic", "verdict": "Causally Suggestive", "failure_modes": [], "shared_nodes": ["mitochondrial"], "upgrade_class": "U3"},
    {"id": "031a", "family": "Bipolar", "claim": "Circadian/lithium mechanism", "verdict": "Causally Suggestive", "failure_modes": [], "shared_nodes": ["circadian"], "upgrade_class": "U1"},
    {"id": "031b", "family": "Bipolar", "claim": "Clock genes as risk genes", "verdict": "Disconfirmed", "failure_modes": ["I1"], "shared_nodes": ["circadian"], "upgrade_class": None},
    {"id": "032", "family": "Bipolar", "claim": "CACNA1C/calcium-channel", "verdict": "Causally Suggestive", "failure_modes": ["C4"], "shared_nodes": [], "upgrade_class": "U1"},
    {"id": "033", "family": "Bipolar", "claim": "Kindling/sensitization", "verdict": "Inconclusive", "failure_modes": ["C1", "I5"], "shared_nodes": [], "upgrade_class": "U4"},
    {"id": "034", "family": "OCD", "claim": "CSTC/habit-circuit", "verdict": "Mechanistically Supported", "failure_modes": ["C4"], "shared_nodes": ["CSTC_circuit"], "upgrade_class": None},
    {"id": "035", "family": "OCD", "claim": "Serotonergic deficit (OCD)", "verdict": "Disconfirmed", "failure_modes": ["F9"], "shared_nodes": ["serotonin"], "upgrade_class": None},
    {"id": "036", "family": "OCD", "claim": "Glutamate hypothesis (OCD)", "verdict": "Underdetermined", "failure_modes": ["M1"], "shared_nodes": ["NMDA_glutamate"], "upgrade_class": "U3"},
    {"id": "037", "family": "ASD", "claim": "E/I imbalance (framework)", "verdict": "Causally Suggestive", "failure_modes": ["I3"], "shared_nodes": ["NMDA_glutamate"], "upgrade_class": "U4"},
    {"id": "037-inst", "family": "ASD", "claim": "E/I specific gene subtypes (SHANK/FMR1)", "verdict": "Mechanistically Supported", "failure_modes": ["C4"], "shared_nodes": ["NMDA_glutamate", "synaptic_pruning"], "upgrade_class": None},
    {"id": "038", "family": "ASD", "claim": "MIA/prenatal immunity", "verdict": "Causally Suggestive", "failure_modes": ["I5", "C4"], "shared_nodes": ["MIA_prenatal", "IL6_inflammation"], "upgrade_class": "U1"},
    {"id": "039", "family": "ASD", "claim": "Synaptic/connectome (high-penetrance)", "verdict": "Mechanistically Supported", "failure_modes": ["C4"], "shared_nodes": ["synaptic_pruning"], "upgrade_class": None},
    {"id": "040", "family": "Addiction", "claim": "Dopamine-RPE reward", "verdict": "Mechanistically Supported", "failure_modes": [], "shared_nodes": ["dopamine"], "upgrade_class": None},
    {"id": "041", "family": "Addiction", "claim": "Allostatic/opponent-process", "verdict": "Causally Suggestive", "failure_modes": ["C1"], "shared_nodes": ["HPA_axis"], "upgrade_class": "U2"},
    {"id": "042", "family": "Addiction", "claim": "Incentive-sensitization", "verdict": "Causally Suggestive", "failure_modes": ["I7"], "shared_nodes": ["dopamine"], "upgrade_class": "U2"},
    {"id": "043", "family": "ADHD", "claim": "Catecholamine/PFC-executive", "verdict": "Causally Suggestive", "failure_modes": ["I3"], "shared_nodes": ["noradrenergic", "dopamine"], "upgrade_class": "U1"},
    {"id": "044", "family": "ADHD", "claim": "Maturational-delay/neurodevelopmental", "verdict": "Causally Suggestive", "failure_modes": ["I1"], "shared_nodes": [], "upgrade_class": "U1"},
]

SHARED_NODES = {
    "FKBP5": {"disorders": ["Depression", "PTSD"], "entries": ["015", "017"]},
    "IL6_inflammation": {"disorders": ["Depression", "Schizophrenia", "ASD"], "entries": ["006a", "029", "038"]},
    "NMDA_glutamate": {"disorders": ["Depression", "Schizophrenia", "OCD", "ASD"], "entries": ["012", "026", "036", "037"]},
    "mitochondrial": {"disorders": ["Depression", "Bipolar"], "entries": ["013", "030"]},
    "circadian": {"disorders": ["Depression", "Bipolar"], "entries": ["014", "031a", "031b"]},
    "fear_circuit": {"disorders": ["PTSD", "Anxiety"], "entries": ["018", "022"]},
    "noradrenergic": {"disorders": ["PTSD", "Anxiety", "ADHD"], "entries": ["019", "024", "043"]},
    "serotonin": {"disorders": ["Depression", "Anxiety", "OCD"], "entries": ["003", "023", "035"]},
    "MIA_prenatal": {"disorders": ["Schizophrenia", "ASD"], "entries": ["029", "038"]},
    "dopamine": {"disorders": ["Schizophrenia", "Addiction", "ADHD"], "entries": ["025", "040", "042", "043"]},
    "synaptic_pruning": {"disorders": ["Schizophrenia", "ASD"], "entries": ["027", "039"]},
    "HPA_axis": {"disorders": ["Depression", "PTSD", "Addiction"], "entries": ["004", "015", "017", "020", "041"]},
    "CSTC_circuit": {"disorders": ["OCD"], "entries": ["034"]},
}

FAILURE_MODES = {
    "C1": "Circularity/unfalsifiability",
    "C2": "Backwards/contested direction",
    "C3": "High-dimensional fragility",
    "C4": "Non-specificity",
    "I1": "Missing/weak causal evidence",
    "I3": "Specificity/hard-baseline failure",
    "I5": "Reverse causation/confounding",
    "I7": "Missing intervention",
    "M1": "Proxy invalidity",
    "M2": "Measurement inconsistency",
    "F9": "Affirming the consequent (treatment->etiology)",
}

FAMILIES = ["Depression", "PTSD", "Anxiety", "Schizophrenia", "Bipolar", "OCD", "ASD", "Addiction", "ADHD"]

GWAS_TO_FAMILY = {
    "adhd2022": "ADHD",
    "anx2026": "Anxiety",
    "asd2019": "ASD",
    "bip2024": "Bipolar",
    "mdd2025": "Depression",
    "ptsd2024": "PTSD",
    "scz2022": "Schizophrenia",
    "sud2023": "Addiction",
}
