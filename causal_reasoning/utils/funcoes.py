from typing import List, Tuple

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


def get_tuple_edges(edges_str: str) -> List[Tuple[str, str]]:
    """
    Extracts the edges in tuple form from the edges string.

    Args:
        edges_str (str): The edges string.

    Returns:
        List[Tuple[str, str]]: A list of tuples with the nodes.
    """
    edges_part = edges_str.split(",")
    edges = []

    for part in edges_part:
        part = part.strip()
        left, right = part.split("->")
        left = left.strip()
        right = right.strip()

        edges.append((left, right))

    return edges


def generate_example(
    edges_str: str,
    latent: list[str],
    intervention: list[str],
    target: str,
    custom_cardinalities: dict = None,
):
    """
    Generates an example graph based on the edges string.

    Args:
        edges (str): The edges string.
        custom_cardinalities (dict, optional): A dictionary with custom cardinalities for the nodes.
    """
    edges = get_tuple_edges(edges_str)
    graph = nx.DiGraph()
    graph.add_edges_from(edges)

    if custom_cardinalities is None:
        custom_cardinalities = {}

    node_colors = define_colors(graph, latent, intervention, target)
    draw_graph(graph, node_colors)


def tuple_generate_example(edges, custom_cardinalities: dict = None):
    """
    Generates an example graph based on the edges string.

    Args:
        edges (str): The edges string.
        custom_cardinalities (dict, optional): A dictionary with custom cardinalities for the nodes.
    """
    graph = nx.DiGraph()
    graph.add_edges_from(edges)

    if custom_cardinalities is None:
        custom_cardinalities = {}

    node_colors = define_colors(graph)
    draw_graph(graph, node_colors)
