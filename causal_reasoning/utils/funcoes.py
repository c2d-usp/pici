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


def str_to_joaos(
    versao_str: str, latent: list[str], custom_cardinalities: dict = None
) -> str:
    """
    Converts a string of edges like:
      "U1 -> X, U1 -> Y, U2 -> Z, X -> Y, Z -> X"
    into the a string that follows the pattern:
      5 # Number of nodes \n
      5 # Number of egdes \n
      U1 0 # Node U1 has cardinality 0 \n
      U2 0 # Node U2 has cardinality 0 \n
      X 2 # Node X has cardinality 2 \n
      Y 2 # Node Y has cardinality 2 \n
      Z 2 # Node Z has cardinality 2 \n
      U1 X # Edge U1 -> X \n
      U1 Y # Edge U1 -> Y \n
      U2 Z # Edge U2 -> Z \n
      X Y # Edge X -> Y \n
      Z X # Edge Z -> X
    """
    if custom_cardinalities is None:
        custom_cardinalities = {}

    edges_part = versao_str.split(",")
    edges = []
    node_order = []
    node_set = set()

    for part in edges_part:
        part = part.strip()
        left, right = part.split("->")
        left = left.strip()
        right = right.strip()

        edges.append((left, right))

        for n in (left, right):
            if n not in node_set:
                node_order.append(n)
                node_set.add(n)

    node_card = {}
    for node in node_order:
        if node in custom_cardinalities:
            node_card[node] = custom_cardinalities[node]
        else:
            node_card[node] = 0 if node in latent else 2

    u_nodes = [n for n in node_order if n in latent]
    other_nodes = [n for n in node_order if n not in latent]
    final_node_order = u_nodes + other_nodes

    lines = []

    lines.append(str(len(final_node_order)))
    lines.append(str(len(edges)))

    for node in final_node_order:
        lines.append(f"{node} {node_card[node]}")

    for left, right in edges:
        lines.append(f"{left} {right}")

    return "\n".join(lines)


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


def get_joaos_input(
    edges_str: str,
    latent: list[str],
    file_path: str = "",
    custom_cardinalities: dict = None,
):
    joaos_str = str_to_joaos(edges_str, latent, custom_cardinalities)
    if not file_path == "":
        f = open(file_path, "w")
        f.write(joaos_str)
    else:
        print(joaos_str)


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
