"""Generate all figures for Paper A: Comorbidity Network Curvature.

Usage:
    uv run --no-project --with numpy --with scipy --with networkx --with pot --with matplotlib --with tqdm --with adjustText \
        python psych/experiments/generate_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx
import ot
from adjustText import adjust_text
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

sys.path.insert(0, str(Path(__file__).parent))
from catalog_data import ENTRIES, SHARED_NODES, FAMILIES, VERDICTS

OUT_DIR = Path("psych/experiments/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAMILY_COLORS = {
    "Depression": "#C0392B",
    "PTSD": "#D35400",
    "Anxiety": "#F39C12",
    "Schizophrenia": "#8E44AD",
    "Bipolar": "#2980B9",
    "OCD": "#16A085",
    "ASD": "#27AE60",
    "Addiction": "#C2185B",
    "ADHD": "#E67E22",
}

MECHANISM_COLOR = "#5D6D7E"


def build_graph():
    G = nx.Graph()
    for fam in FAMILIES:
        G.add_node(f"family:{fam}", layer="family", label=fam)

    for node_name, info in SHARED_NODES.items():
        label = (node_name.replace("_", " ")
                 .replace("IL6 inflammation", "IL-6")
                 .replace("NMDA glutamate", "NMDA/Glu")
                 .replace("MIA prenatal", "MIA/prenatal")
                 .replace("synaptic pruning", "Syn. pruning")
                 .replace("fear circuit", "Fear circuit")
                 .replace("CSTC circuit", "CSTC"))
        G.add_node(f"mechanism:{node_name}", layer="mechanism",
                   n_disorders=len(info["disorders"]), label=label)
        for disorder in info["disorders"]:
            G.add_edge(f"family:{disorder}", f"mechanism:{node_name}")

    for entry in ENTRIES:
        eid = f"claim:{entry['id']}"
        G.add_node(eid, layer="claim",
                   verdict=entry["verdict"],
                   verdict_score=VERDICTS.get(entry["verdict"], 2),
                   family=entry["family"],
                   label=entry["id"])
        G.add_edge(eid, f"family:{entry['family']}")
        for node in entry["shared_nodes"]:
            if f"mechanism:{node}" in G:
                G.add_edge(eid, f"mechanism:{node}")
    return G


def build_bipartite_graph():
    """Build the 22-node bipartite graph (9 disorders + 13 mechanisms only)."""
    G = nx.Graph()
    for fam in FAMILIES:
        G.add_node(f"family:{fam}", layer="family", label=fam)
    for node_name, info in SHARED_NODES.items():
        label = (node_name.replace("_", " ")
                 .replace("IL6 inflammation", "IL-6")
                 .replace("NMDA glutamate", "NMDA/Glu")
                 .replace("MIA prenatal", "MIA/prenatal")
                 .replace("synaptic pruning", "Syn. pruning")
                 .replace("fear circuit", "Fear circuit")
                 .replace("CSTC circuit", "CSTC"))
        G.add_node(f"mechanism:{node_name}", layer="mechanism",
                   n_disorders=len(info["disorders"]), label=label)
        for disorder in info["disorders"]:
            G.add_edge(f"family:{disorder}", f"mechanism:{node_name}")
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


def fig1_network_diagram(G_full, curvatures_full):
    """Bipartite network: 9 disorders + 13 mechanisms, curvature-colored edges."""
    G = build_bipartite_graph()
    curvatures = compute_curvatures(G)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    family_nodes = sorted([n for n in G if n.startswith("family:")],
                          key=lambda n: -G.degree(n))
    mech_nodes = sorted([n for n in G if n.startswith("mechanism:")],
                        key=lambda n: -G.degree(n))

    pos = {}
    for i, n in enumerate(family_nodes):
        pos[n] = (-1.2, 1.0 - i * (2.0 / (len(family_nodes) - 1)))
    for i, n in enumerate(mech_nodes):
        pos[n] = (1.2, 1.0 - i * (2.0 / (len(mech_nodes) - 1)))

    curv_vals = list(curvatures.values())
    vabs = max(abs(min(curv_vals)), abs(max(curv_vals)))
    norm = Normalize(vmin=-vabs, vmax=vabs)
    cmap = plt.cm.RdBu

    for u, v in G.edges():
        c = curvatures.get((u, v), curvatures.get((v, u), 0))
        color = cmap(norm(c))
        lw = 1.0 + abs(c) * 6
        alpha = 0.4 + abs(c) * 1.5
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=color, linewidth=lw, alpha=min(alpha, 0.9), zorder=1,
                solid_capstyle="round")

    for n in mech_nodes:
        label = G.nodes[n].get("label", n.split(":")[1])
        deg = G.degree(n)
        size = 80 + deg * 40
        ax.scatter(*pos[n], s=size, c=MECHANISM_COLOR, edgecolors="black",
                   linewidths=1.5, zorder=3, marker="o")
        ax.annotate(label, pos[n], fontsize=8, ha="left", va="center",
                    xytext=(12, 0), textcoords="offset points",
                    fontweight="bold", color="#2C3E50")

    for n in family_nodes:
        fam = n.split(":")[1]
        color = FAMILY_COLORS.get(fam, "#999999")
        deg = G.degree(n)
        size = 120 + deg * 25
        ax.scatter(*pos[n], s=size, c=color, edgecolors="black",
                   linewidths=2, zorder=3, marker="s")
        ax.annotate(fam, pos[n], fontsize=9, ha="right", va="center",
                    xytext=(-12, 0), textcoords="offset points",
                    fontweight="bold", color="#2C3E50")

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02, aspect=30)
    cbar.set_label("Ollivier-Ricci curvature", fontsize=10)

    ax.set_xlim(-2.2, 2.6)
    ax.set_ylim(-1.15, 1.15)
    ax.set_title("Disorder-Mechanism Network", fontsize=13, fontweight="bold")
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_network_curvature.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig1_network_curvature.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig 1: {OUT_DIR / 'fig1_network_curvature.png'}")


def fig2_degree_vs_curvature(G, curvatures):
    """Node curvature vs degree scatter with repelled labels."""
    null_path = Path("results/psych/psych/curvature_null_model")
    null_files = sorted(null_path.glob("*.json"))
    null_data = None
    if null_files:
        with open(null_files[-1]) as f:
            null_data = json.load(f)

    node_results = {}
    if null_data:
        node_results = null_data.get("node_results", null_data.get("mechanism_results", {}))

    mech_nodes = [n for n in G if n.startswith("mechanism:")]
    node_curvatures = {}
    for n in mech_nodes:
        edges = [(u, v) for u, v in G.edges(n)]
        curvs = []
        for u, v in edges:
            c = curvatures.get((u, v), curvatures.get((v, u), None))
            if c is not None:
                curvs.append(c)
        if curvs:
            node_curvatures[n] = np.mean(curvs)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    degrees = []
    curvs = []
    labels = []
    zscores = []
    for n in mech_nodes:
        if n in node_curvatures:
            degrees.append(G.degree(n))
            curvs.append(node_curvatures[n])
            labels.append(G.nodes[n].get("label", n.split(":")[1]))
            mech_key = n
            info = node_results.get(mech_key, node_results.get(n.split(":")[1], {}))
            zscores.append(info.get("z_score", 0))

    sig_colors = []
    for z in zscores:
        if abs(z) > 1.96:
            sig_colors.append("#E74C3C" if z < 0 else "#2ECC71")
        else:
            sig_colors.append("#3498DB")

    ax.scatter(degrees, curvs, s=100, c=sig_colors, edgecolors="black",
               linewidths=0.8, zorder=3)

    # Tight hand-placed offsets — just enough to clear the dot
    offsets = {
        "CSTC": (6, 4),
        "MIA/prenatal": (6, 4),
        "Syn. pruning": (6, -10),
        "Fear circuit": (-58, -10),
        "FKBP5": (6, 6),
        "circadian": (-52, -10),
        "mitochondrial": (6, -10),
        "IL-6": (6, 4),
        "HPA axis": (6, 4),
        "dopamine": (6, -10),
        "noradrenergic": (-70, -10),
        "serotonin": (-52, -10),
        "NMDA/Glu": (6, 4),
    }
    for i, label in enumerate(labels):
        dx, dy = offsets.get(label, (6, 0))
        ax.annotate(label, (degrees[i], curvs[i]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=7.5, fontweight="bold", color="#2C3E50",
                    zorder=5)

    z = np.polyfit(degrees, curvs, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(degrees) - 0.5, max(degrees) + 0.5, 100)
    ax.plot(x_line, p(x_line), "--", color="#95A5A6", linewidth=1.5,
            label="$r = -0.73$, $p = 0.005$")

    ax.axhline(0, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
    ax.set_xlabel("Node degree", fontsize=11)
    ax.set_ylabel("Mean Ollivier-Ricci curvature", fontsize=11)

    sig_patch = mpatches.Patch(color="#2ECC71", label="$|z| > 1.96$ (isolated)")
    ns_patch = mpatches.Patch(color="#3498DB", label="$|z| \\leq 1.96$")
    ax.legend(handles=[sig_patch, ns_patch,
              plt.Line2D([0], [0], linestyle="--", color="#95A5A6", label="$r = -0.73$, $p = 0.005$")],
              fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_degree_vs_curvature.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig2_degree_vs_curvature.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig 2: {OUT_DIR / 'fig2_degree_vs_curvature.png'}")


def fig3_edge_zscore_heatmap():
    """Edge-level z-score heatmap (disorder x mechanism)."""
    edge_path = Path("results/psych/psych/edge_curvature_null")
    edge_files = sorted(edge_path.glob("*.json"))
    if not edge_files:
        print("  Fig 3: SKIP — no edge curvature results")
        return

    with open(edge_files[-1]) as f:
        data = json.load(f)

    families_ordered = ["Depression", "PTSD", "Anxiety", "Schizophrenia",
                        "Bipolar", "OCD", "ASD", "Addiction", "ADHD"]
    mechs_ordered = ["HPA_axis", "serotonin", "circadian", "NMDA_glutamate", "dopamine",
                     "noradrenergic", "IL6_inflammation", "FKBP5", "mitochondrial",
                     "fear_circuit", "MIA_prenatal", "synaptic_pruning", "CSTC_circuit"]

    matrix = np.full((len(families_ordered), len(mechs_ordered)), np.nan)
    has_edge = np.full((len(families_ordered), len(mechs_ordered)), False)

    for edge_key, info in data.get("edges", {}).items():
        u = info.get("u", "")
        v = info.get("v", "")

        fam = mech = None
        for node in [u, v]:
            if node.startswith("family:"):
                fam = node.replace("family:", "")
            elif node.startswith("mechanism:"):
                mech = node.replace("mechanism:", "")

        if fam and mech and fam in families_ordered and mech in mechs_ordered:
            i = families_ordered.index(fam)
            j = mechs_ordered.index(mech)
            z = info.get("z_score", 0)
            has_edge[i, j] = True
            if np.isnan(matrix[i, j]) or abs(z) > abs(matrix[i, j]):
                matrix[i, j] = z

    fig, ax = plt.subplots(1, 1, figsize=(11, 6))

    vmax = max(3.5, np.nanmax(np.abs(matrix[np.isfinite(matrix)])))
    cmap = plt.cm.RdBu
    im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    for i in range(len(families_ordered)):
        for j in range(len(mechs_ordered)):
            if not has_edge[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             facecolor="#F0F0F0", edgecolor="#E0E0E0", linewidth=0.5))

    mech_labels = []
    for m in mechs_ordered:
        label = (m.replace("_", " ")
                 .replace("IL6 inflammation", "IL-6")
                 .replace("NMDA glutamate", "NMDA/Glu")
                 .replace("MIA prenatal", "MIA/prenatal")
                 .replace("synaptic pruning", "Syn. pruning")
                 .replace("fear circuit", "Fear circuit")
                 .replace("CSTC circuit", "CSTC"))
        mech_labels.append(label)

    ax.set_xticks(range(len(mechs_ordered)))
    ax.set_xticklabels(mech_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(families_ordered)))
    ax.set_yticklabels(families_ordered, fontsize=10)

    for i in range(len(families_ordered)):
        for j in range(len(mechs_ordered)):
            val = matrix[i, j]
            if np.isfinite(val):
                sig = "***" if abs(val) > 3 else "**" if abs(val) > 2 else "*" if abs(val) > 1.96 else ""
                color = "white" if abs(val) > 2.5 else "black"
                fontsize = 8 if sig else 7
                ax.text(j, i, f"{val:+.1f}{sig}", ha="center", va="center",
                        fontsize=fontsize, color=color,
                        fontweight="bold" if sig else "normal")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("$z$-score (vs degree-preserving null)", fontsize=10)

    ax.set_title("Edge-level curvature $z$-scores: disorder $\\times$ mechanism",
                 fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_edge_zscore_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig3_edge_zscore_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig 3: {OUT_DIR / 'fig3_edge_zscore_heatmap.png'}")


def fig4_mr_heatmap():
    """MR beta heatmap (8x8 disorder matrix)."""
    mr_path = Path("results/psych/psych/cross_disorder_mr_v2")
    mr_files = sorted(mr_path.glob("mr_redo_v5*.json"))
    if not mr_files:
        mr_files = sorted(mr_path.glob("mr_redo_v4*.json"))
    if not mr_files:
        mr_files = sorted(mr_path.glob("mr_redo_v3*.json"))
    if not mr_files:
        mr_files = sorted(mr_path.glob("*.json"))
    if not mr_files:
        print("  Fig 4: SKIP — no MR results")
        return

    with open(mr_files[-1]) as f:
        data = json.load(f)

    results = data.get("mr_results", data.get("results", {}))
    disorders = ["adhd2022", "anx2026", "asd2019", "bip2024",
                 "mdd2025", "ptsd2024", "scz2022", "sud2023"]
    labels = ["ADHD", "ANX", "ASD", "BIP", "MDD", "PTSD", "SCZ", "SUD"]

    matrix = np.full((len(disorders), len(disorders)), np.nan)
    has_result = np.full((len(disorders), len(disorders)), False)

    for key, val in results.items():
        parts = key.split("->")
        if len(parts) != 2:
            continue
        exp, out = parts
        if exp in disorders and out in disorders and val.get("status") == "success":
            i = disorders.index(exp)
            j = disorders.index(out)
            ivw = val.get("ivw", {})
            beta = ivw.get("beta", None)
            if beta is not None:
                matrix[i, j] = float(beta)
                has_result[i, j] = True

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6))

    display_matrix = np.copy(matrix)
    display_matrix = np.clip(display_matrix, 0, 1.0)

    colors_list = ["#FFFFCC", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#B10026"]
    cmap = LinearSegmentedColormap.from_list("mr_sequential", colors_list, N=256)

    im = ax.imshow(display_matrix, cmap=cmap, vmin=0, vmax=0.8, aspect="equal")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_xlabel("Outcome", fontsize=11)
    ax.set_ylabel("Exposure", fontsize=11)
    ax.xaxis.set_label_position("bottom")

    for i in range(len(labels)):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                    fill=True, facecolor="#F5F5F5", edgecolor="#CCCCCC",
                                    linewidth=0.5))

    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                continue
            val = matrix[i, j]
            if np.isfinite(val):
                display_val = val
                color = "white" if display_val > 0.5 else "black"
                ax.text(j, i, f"{display_val:.2f}", ha="center", va="center",
                        fontsize=7.5, color=color, fontweight="bold" if display_val > 0.5 else "normal")
            elif not has_result[i, j]:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=9, color="#AAAAAA")
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             fill=True, facecolor="#F5F5F5", edgecolor="#E0E0E0",
                             linewidth=0.5))

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("IVW $\\beta$ (LD-clumped)", fontsize=10)

    ax.set_title("Cross-disorder Mendelian randomization", fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_mr_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig4_mr_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig 4: {OUT_DIR / 'fig4_mr_heatmap.png'}")


def fig5_node_null_distributions():
    """Node-level null model: observed vs null distributions for key mechanisms."""
    null_path = Path("results/psych/psych/curvature_null_model")
    null_files = sorted(null_path.glob("*.json"))
    if not null_files:
        print("  Fig 5: SKIP — no null model results")
        return

    with open(null_files[-1]) as f:
        data = json.load(f)

    mechs_of_interest = ["serotonin", "HPA_axis", "MIA_prenatal",
                         "synaptic_pruning", "CSTC_circuit", "circadian"]
    mech_labels = {
        "serotonin": "Serotonin",
        "HPA_axis": "HPA axis",
        "MIA_prenatal": "MIA/prenatal",
        "synaptic_pruning": "Syn. pruning",
        "CSTC_circuit": "CSTC circuit",
        "circadian": "Circadian",
    }

    node_results = data.get("node_results", data.get("mechanism_results", {}))

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))

    for idx, mech in enumerate(mechs_of_interest):
        ax = axes[idx // 3][idx % 3]

        key = f"mechanism:{mech}"
        info = node_results.get(key, node_results.get(mech, None))
        if info is None:
            ax.text(0.5, 0.5, f"No data: {mech}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
            continue

        obs = info.get("observed", info.get("observed_curvature", 0))
        null_mean = info.get("null_mean", 0)
        null_std = info.get("null_std", 0.1)
        z = info.get("z_score", 0)

        x = np.linspace(null_mean - 4.5 * null_std, null_mean + 4.5 * null_std, 300)
        pdf = (1 / (null_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - null_mean) / null_std) ** 2)

        reject_lo = null_mean - 1.96 * null_std
        reject_hi = null_mean + 1.96 * null_std

        ax.fill_between(x, pdf, alpha=0.15, color="#3498DB")
        ax.plot(x, pdf, color="#3498DB", linewidth=1.5)

        x_lo = x[x <= reject_lo]
        pdf_lo = pdf[:len(x_lo)]
        if len(x_lo) > 0:
            ax.fill_between(x_lo, pdf_lo, alpha=0.25, color="#E74C3C")
        x_hi = x[x >= reject_hi]
        pdf_hi = pdf[len(pdf) - len(x_hi):]
        if len(x_hi) > 0:
            ax.fill_between(x_hi, pdf_hi, alpha=0.25, color="#E74C3C")

        ax.axvline(obs, color="#C0392B", linewidth=2.5, linestyle="-", zorder=5)

        label = mech_labels.get(mech, mech)
        sig_marker = "*" if abs(z) > 1.96 else ""
        ax.set_title(f"{label}  ($z = {z:+.2f}${sig_marker})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Curvature", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.tick_params(labelsize=8)

        obs_txt = f"obs = {obs:.3f}"
        ax.annotate(obs_txt, xy=(obs, ax.get_ylim()[1] * 0.85),
                    fontsize=6.5, color="#C0392B", ha="left" if z > 0 else "right",
                    xytext=(5 if z > 0 else -5, 0), textcoords="offset points")

    fig.suptitle("Node-level curvature: observed vs degree-preserving null (1,000 permutations)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_node_null_distributions.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig5_node_null_distributions.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig 5: {OUT_DIR / 'fig5_node_null_distributions.png'}")


def fig6_verdict_curvature():
    """Verdict tier vs mean curvature for claims."""
    G = build_graph()
    curvatures = compute_curvatures(G)

    verdict_order = ["Disconfirmed", "Underdetermined", "Inconclusive",
                     "Causally Suggestive", "Mechanistically Supported", "Validated"]
    short_labels = ["Disconf.", "Undet.", "Inconc.", "Causal. Sug.", "Mech. Sup.", "Validated"]

    claim_data = []
    for entry in ENTRIES:
        eid = f"claim:{entry['id']}"
        edges = [(u, v) for u, v in G.edges(eid)]
        curvs = []
        for u, v in edges:
            c = curvatures.get((u, v), curvatures.get((v, u), None))
            if c is not None:
                curvs.append(c)
        if curvs:
            claim_data.append({
                "verdict": entry["verdict"],
                "curvature": np.mean(curvs),
            })

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    by_verdict = {}
    for d in claim_data:
        v = d["verdict"]
        by_verdict.setdefault(v, []).append(d["curvature"])

    gradient_colors = ["#E8D5B7", "#D4B896", "#BFA87A", "#9BBB59", "#548235", "#2E7D32"]

    positions = []
    means = []
    stds = []
    colors = []
    used_labels = []
    for i, v in enumerate(verdict_order):
        if v in by_verdict:
            vals = by_verdict[v]
            positions.append(len(positions))
            means.append(np.mean(vals))
            stds.append(np.std(vals) / np.sqrt(len(vals)))
            colors.append(gradient_colors[i])
            used_labels.append(short_labels[i])

    bars = ax.bar(positions, means, yerr=stds, capsize=4, color=colors,
                  edgecolor="black", linewidth=0.6, width=0.7)

    for bar_idx, (pos, m, n_label) in enumerate(zip(positions, means, used_labels)):
        v = verdict_order[[sl for sl in range(len(short_labels)) if short_labels[sl] == n_label][0]]
        n = len(by_verdict.get(v, []))
        ax.text(pos, m + stds[bar_idx] + 0.005, f"$n={n}$", ha="center",
                va="bottom", fontsize=9, color="#222222", fontweight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels(used_labels, fontsize=9, rotation=0, ha="center")
    ax.set_ylabel("Mean curvature ($\\pm$ SE)", fontsize=10)
    ax.set_ylim(-0.01, 0.18)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)

    ax.annotate("$r = 0.257$, $p = 0.084$ (partial, controlling degree)",
                xy=(0.5, 1.02), xycoords="axes fraction", fontsize=8,
                ha="center", color="#555555")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_verdict_curvature.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig6_verdict_curvature.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig 6: {OUT_DIR / 'fig6_verdict_curvature.png'}")


def figS1_full_network(G, curvatures):
    """Supplementary: full 69-node graph with all claim nodes."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    family_nodes = sorted([n for n in G if G.nodes[n].get("layer") == "family"],
                          key=lambda n: -G.degree(n))
    mech_nodes = sorted([n for n in G if G.nodes[n].get("layer") == "mechanism"],
                        key=lambda n: -G.degree(n))
    claim_nodes = sorted([n for n in G if G.nodes[n].get("layer") == "claim"],
                         key=lambda n: G.nodes[n].get("family", ""))

    pos = {}
    for i, n in enumerate(family_nodes):
        pos[n] = (-2.0, 1.0 - i * (2.0 / max(len(family_nodes) - 1, 1)))
    for i, n in enumerate(mech_nodes):
        pos[n] = (0.0, 1.0 - i * (2.0 / max(len(mech_nodes) - 1, 1)))
    for i, n in enumerate(claim_nodes):
        pos[n] = (2.0, 1.0 - i * (2.0 / max(len(claim_nodes) - 1, 1)))

    curv_vals = list(curvatures.values())
    vabs = max(abs(min(curv_vals)), abs(max(curv_vals)))
    norm = Normalize(vmin=-vabs, vmax=vabs)
    cmap = plt.cm.RdBu

    for u, v in G.edges():
        c = curvatures.get((u, v), curvatures.get((v, u), 0))
        color = cmap(norm(c))
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=color, linewidth=0.5, alpha=0.4, zorder=1)

    for n in claim_nodes:
        fam = G.nodes[n].get("family", "")
        color = FAMILY_COLORS.get(fam, "#CCCCCC")
        ax.scatter(*pos[n], s=15, c=color, edgecolors="none", zorder=2, alpha=0.7)

    for n in mech_nodes:
        label = G.nodes[n].get("label", n.split(":")[1])
        deg = G.degree(n)
        size = 60 + deg * 20
        ax.scatter(*pos[n], s=size, c=MECHANISM_COLOR, edgecolors="black",
                   linewidths=1, zorder=3)
        ax.annotate(label, pos[n], fontsize=6, ha="left", va="center",
                    xytext=(8, 0), textcoords="offset points", color="#2C3E50")

    for n in family_nodes:
        fam = n.split(":")[1]
        color = FAMILY_COLORS.get(fam, "#999999")
        ax.scatter(*pos[n], s=100, c=color, edgecolors="black",
                   linewidths=1.5, zorder=3, marker="s")
        ax.annotate(fam, pos[n], fontsize=7, ha="right", va="center",
                    xytext=(-10, 0), textcoords="offset points",
                    fontweight="bold", color="#2C3E50")

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Ollivier-Ricci curvature", fontsize=9)

    ax.annotate("Disorders", xy=(-2.0, 1.08), fontsize=10, ha="center",
                fontweight="bold", color="#2C3E50")
    ax.annotate("Mechanisms", xy=(0.0, 1.08), fontsize=10, ha="center",
                fontweight="bold", color="#2C3E50")
    ax.annotate("Claims (n=47)", xy=(2.0, 1.08), fontsize=10, ha="center",
                fontweight="bold", color="#2C3E50")

    ax.set_xlim(-3.0, 3.2)
    ax.set_ylim(-1.15, 1.15)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figS1_full_network.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "figS1_full_network.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig S1: {OUT_DIR / 'figS1_full_network.png'}")


def figS2_alpha_sensitivity(G):
    """Supplementary: curvature ranking stability across alpha values."""
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]

    mech_nodes = sorted([n for n in G if G.nodes[n].get("layer") == "mechanism"])

    all_curvatures = {}
    for alpha in alphas:
        curvs = compute_curvatures(G, alpha=alpha)
        node_curvs = {}
        for n in mech_nodes:
            edges = [(u, v) for u, v in G.edges(n)]
            vals = []
            for u, v in edges:
                c = curvs.get((u, v), curvs.get((v, u), None))
                if c is not None:
                    vals.append(c)
            if vals:
                node_curvs[n] = np.mean(vals)
        all_curvatures[alpha] = node_curvs

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for n in mech_nodes:
        label = G.nodes[n].get("label", n.split(":")[1])
        vals = [all_curvatures[a].get(n, np.nan) for a in alphas]
        if any(np.isfinite(v) for v in vals):
            ax.plot(alphas, vals, "o-", markersize=4, label=label, linewidth=1.2)

    ax.set_xlabel("Idleness parameter $\\alpha$", fontsize=11)
    ax.set_ylabel("Mean node curvature", fontsize=11)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.4)
    ax.legend(fontsize=6.5, ncol=2, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figS2_alpha_sensitivity.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "figS2_alpha_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig S2: {OUT_DIR / 'figS2_alpha_sensitivity.png'}")


def main():
    print("Building graph and computing curvatures...")
    G = build_graph()
    curvatures = compute_curvatures(G)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(curvatures)} edge curvatures")

    print("\nGenerating figures...")
    fig1_network_diagram(G, curvatures)
    fig2_degree_vs_curvature(G, curvatures)
    fig3_edge_zscore_heatmap()
    fig4_mr_heatmap()
    fig5_node_null_distributions()
    fig6_verdict_curvature()

    print("\nGenerating supplementary figures...")
    figS1_full_network(G, curvatures)
    figS2_alpha_sensitivity(G)
    print("\nDone.")


if __name__ == "__main__":
    main()
