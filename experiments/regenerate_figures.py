"""Regenerate figures for V5 with improved visuals.

- Fig 1: Network with external labels, red-blue diverging edge colors (like psych paper)
- Fig 2: Slope graph (kept as-is but minor polish)
- Fig 3: Bootstrap CI (kept as-is)
"""
import json
import numpy as np
import networkx as nx
import ot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

EDGES = [
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

DISORDERS = {"MDD", "GAD", "SUD", "PTSD", "SCZ", "BIP", "BPD", "OCD", "ADHD", "ASD"}
SYMPTOMS = {"insomnia", "anhedonia", "suicide_ideation", "fatigue", "cortisol",
            "inflammation", "psychosis", "dissociation", "panic", "agoraphobia",
            "social_anxiety", "avoidance", "somatic"}

DISPLAY_NAMES = {
    "MDD": "MDD", "GAD": "GAD", "SUD": "SUD", "PTSD": "PTSD",
    "SCZ": "SCZ", "BIP": "Bipolar", "BPD": "BPD", "OCD": "OCD",
    "ADHD": "ADHD", "ASD": "ASD",
    "insomnia": "Insomnia", "anhedonia": "Anhedonia",
    "suicide_ideation": "Suicidal ideation", "fatigue": "Fatigue",
    "cortisol": "Cortisol", "inflammation": "Inflammation",
    "psychosis": "Psychosis", "dissociation": "Dissociation",
    "panic": "Panic", "agoraphobia": "Agoraphobia",
    "social_anxiety": "Social anxiety", "avoidance": "Avoidance",
    "somatic": "Somatic",
}

def build_graph():
    G = nx.Graph()
    for a, b in EDGES:
        G.add_edge(a, b)
    return G

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
    for u, v in G.edges():
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
        result[v] = np.mean([curvatures[(a, b)] for a, b in edges])
    return result

def fig_network():
    """Spring layout network with external labels and diverging red-blue edges."""
    G = build_graph()
    orc = compute_orc(G, alpha=0.5)
    mc = mean_curvature(G, orc)

    # Spring layout with high k for spacing, deterministic seed
    pos = nx.spring_layout(G, seed=42, k=2.2, iterations=200)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Diverging colormap: red (negative) to blue (positive)
    cmap = plt.cm.RdBu
    norm = mcolors.TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5)

    # Draw edges with curvature coloring
    for u, v in G.edges():
        k = orc.get((u, v), orc.get((v, u), 0))
        color = cmap(norm(k))
        width = 1.0 + 3.5 * abs(k)
        alpha_val = 0.35 + 0.55 * min(abs(k) / 0.4, 1.0)
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=width, alpha=alpha_val,
                solid_capstyle='round', zorder=1)

    # Draw nodes
    for v in G.nodes():
        x, y = pos[v]
        deg = G.degree(v)
        size = 150 + deg * 70
        is_disorder = v in DISORDERS
        node_color = cmap(norm(mc[v]))
        marker = 's' if is_disorder else 'o'
        ax.scatter(x, y, s=size, c=[node_color], marker=marker,
                   edgecolors='black', linewidths=1.2, zorder=3)

    # External labels: place outside nodes, away from graph center
    cx = np.mean([p[0] for p in pos.values()])
    cy = np.mean([p[1] for p in pos.values()])
    for v in G.nodes():
        x, y = pos[v]
        name = DISPLAY_NAMES.get(v, v)
        # Push label away from center
        dx = x - cx
        dy = y - cy
        mag = max(np.sqrt(dx**2 + dy**2), 0.01)
        offset_x = dx / mag * 18
        offset_y = dy / mag * 12
        ha = 'left' if dx > 0 else 'right'
        if abs(dx) < 0.05:
            ha = 'center'
        ax.annotate(name, (x, y), fontsize=8.5, fontweight='bold',
                    ha=ha, va='center',
                    xytext=(offset_x, offset_y), textcoords='offset points')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label('Ollivier-Ricci curvature', fontsize=10)

    # Legend for node shapes
    ax.scatter([], [], s=100, marker='s', c='white', edgecolors='black',
               linewidths=1, label='Disorder')
    ax.scatter([], [], s=100, marker='o', c='white', edgecolors='black',
               linewidths=1, label='Feature/symptom')
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9)

    ax.axis('off')
    ax.set_title('Comorbidity network colored by Ollivier-Ricci curvature',
                 fontsize=13, pad=15)

    plt.tight_layout()
    plt.savefig('figures/fig_network.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig_network.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved fig_network")


def fig_slopegraph():
    """Slope graph: betweenness rank -> curvature rank."""
    G = build_graph()
    orc = compute_orc(G, alpha=0.5)
    mc = mean_curvature(G, orc)
    btwn = nx.betweenness_centrality(G)

    def rank_nodes(scores, ascending=True):
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=not ascending)
        return {node: rank + 1 for rank, (node, _) in enumerate(sorted_nodes)}

    orc_rank = rank_nodes(mc, ascending=True)
    btwn_rank = rank_nodes(btwn, ascending=False)
    n_nodes = len(G.nodes())

    fig, ax = plt.subplots(1, 1, figsize=(6, 8))

    highlight = {
        "OCD": '#d62728', "GAD": '#1f77b4', "MDD": '#ff7f0e',
        "somatic": '#9467bd', "BIP": '#8c564b', "insomnia": '#e377c2',
    }

    for v in G.nodes():
        br = btwn_rank[v]
        cr = orc_rank[v]
        delta = br - cr
        color = highlight.get(v, '#cccccc')
        lw = 2.5 if v in highlight else 0.5
        alpha_val = 1.0 if v in highlight else 0.3
        ax.plot([0, 1], [br, cr], color=color, linewidth=lw, alpha=alpha_val, zorder=2 if v in highlight else 1)

        if v in highlight:
            name = DISPLAY_NAMES.get(v, v)
            ax.annotate(name, xy=(0, br), fontsize=9, ha='right', va='center',
                        fontweight='bold', color=color, xytext=(-8, 0),
                        textcoords='offset points')
            if abs(delta) >= 3:
                label = f'{name} ({delta:+d})'
            else:
                label = name
            ax.annotate(label, xy=(1, cr), fontsize=9, ha='left', va='center',
                        fontweight='bold', color=color, xytext=(8, 0),
                        textcoords='offset points')

    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(n_nodes + 0.5, 0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Betweenness\nrank', 'Curvature\nbridge rank'], fontsize=11)
    ax.set_ylabel('Rank (1 = most important)', fontsize=11)
    ax.set_title('Rank changes: betweenness vs. curvature', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('figures/fig_slopegraph.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig_slopegraph.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved fig_slopegraph")


def fig_bootstrap():
    """Bootstrap CI forest plot — simple categorical colors."""
    results = json.load(open('results/reviewer/reviewer_results.json'))
    bc = results['bootstrap_ci']

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    nodes_sorted = sorted(bc.keys(), key=lambda v: bc[v]["mean_kappa"])

    for i, v in enumerate(nodes_sorted):
        d = bc[v]
        mk = d["mean_kappa"]
        if mk < -0.05:
            color = '#d62728'  # red — bridge
        elif mk < 0.05:
            color = '#ff7f0e'  # orange — mixed
        else:
            color = '#2ca02c'  # green — cluster-internal
        ax.errorbar(mk, i,
                     xerr=[[mk - d["ci_lower"]], [d["ci_upper"] - mk]],
                     fmt='o', color=color, capsize=4, markersize=8, markeredgecolor='black',
                     markeredgewidth=0.8, elinewidth=2.0)

    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_yticks(list(range(len(nodes_sorted))))
    ax.set_yticklabels([DISPLAY_NAMES.get(v, v) for v in nodes_sorted], fontsize=10)
    ax.set_xlabel('Mean incident curvature $\\bar{\\kappa}$\n(95% CI from 1,000 perturbation trials)', fontsize=11)
    ax.set_title('Bootstrap confidence intervals on $\\bar{\\kappa}$', fontsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728',
               markeredgecolor='black', markersize=9, label='Bridge ($\\bar{\\kappa} < -0.05$)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e',
               markeredgecolor='black', markersize=9, label='Mixed'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
               markeredgecolor='black', markersize=9, label='Cluster-internal ($\\bar{\\kappa} > 0.05$)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig('figures/fig_bootstrap_ci.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig_bootstrap_ci.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved fig_bootstrap_ci")


if __name__ == "__main__":
    fig_network()
    fig_slopegraph()
    fig_bootstrap()
    print("All figures regenerated.")
