import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

from causal_reasoning.graph.node import Node
from causal_reasoning.utils._enum import PlotGraphColors


def plot_graph_image(
    graph: nx.Graph,
    unobservables: list[Node],
    interventions: list[Node],
    targets: list[Node],
    output_path: str,
) -> None:
    node_labels = [str(node) for node in graph.nodes()]
    unobservables_labels = [node.label for node in unobservables]
    interventions_labels = [node.label for node in interventions]
    targets_labels = [node.label for node in targets]

    node_colors = define_node_colors(
        node_labels, unobservables_labels, interventions_labels, targets_labels
    )

    legend_labels = {
        "Interventions": PlotGraphColors.INTERVENTIONS.value,
        "Targets": PlotGraphColors.TARGETS.value,
        "Unobservables": PlotGraphColors.UNOBSERVABLES.value,
        "Observables": PlotGraphColors.OBSERVABLES.value,
    }

    plt.figure(figsize=(8, 6))

    nx.draw_networkx(
        graph,
        pos=nx.shell_layout(graph),
        with_labels=True,
        node_color=node_colors,
        edge_color="gray",
        node_size=2000,
        font_size=12,
        arrowsize=20,
    )

    if legend_labels:
        legend_handles = [
            mpatches.Patch(color=color, label=label)
            for label, color in legend_labels.items()
        ]
        plt.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0.0,
            frameon=True,
        )

    plt.tight_layout()
    plt.savefig(output_path, format="png", bbox_inches="tight")
    plt.close()


def define_node_colors(
    node_labels: list[str],
    unobservables_labels: list[str],
    interventions_labels: list[str],
    targets_labels: list[str],
) -> list:
    node_colors = []
    for node in node_labels:
        if node in interventions_labels:
            node_colors.append(PlotGraphColors.INTERVENTIONS.value)
        elif node in targets_labels:
            node_colors.append(PlotGraphColors.TARGETS.value)
        elif node in unobservables_labels:
            node_colors.append(PlotGraphColors.UNOBSERVABLES.value)
        else:
            node_colors.append(PlotGraphColors.OBSERVABLES.value)
    return node_colors
