import networkx as nx


def define_colors(
    graph: nx.Graph, latent: list[str], intervention: list[str], target: str
) -> list:
    """
    Define the color of each node in the graph.

    Args:
        graph (nx.Graph): The graph to be colored.

    Returns:
        list: A list of colors, one for each node in the graph.
    """
    node_colors = []
    for node in graph.nodes():
        if node in intervention:
            node_colors.append("yellow")
        elif node == target:
            node_colors.append("orange")
        elif node in latent:
            node_colors.append("lightgray")
        else:
            node_colors.append("lightblue")
    return node_colors


def draw_graph(graph: nx.Graph, node_colors: list = None):
    """
    Draw the graph using networkx.

    Args:
        graph (nx.Graph): The graph to be drawn.
        node_colors (list, optional): The color of each node in the graph.
    """
    nx.draw_networkx(
        graph,
        pos=nx.shell_layout(graph),  # Position of nodes
        with_labels=True,  # Show labels on nodes
        node_color=node_colors,  # Color of nodes
        edge_color="gray",  # Color of edges
        node_size=2000,  # Size of nodes
        font_size=12,  # Font size for labels
        arrowsize=20,  # Arrow size for edges
    )


