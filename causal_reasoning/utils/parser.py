from typing import Any

import networkx as nx

from causal_reasoning.graph.graph import Graph
from causal_reasoning.graph.node import Node


def parse_to_string_list(state):
    if isinstance(state, str):
        return [state]
    if isinstance(state, int):
        return [str(state)]
    if isinstance(state, list):
        if all(isinstance(item, str) for item in state):
            return state
        for item in state:
            if isinstance(item, int):
                item = str(item)
            if not isinstance(item, str):
                raise Exception(f"Input format for {state} not recognized.")
        return state
    raise Exception(f"Input format for {state} not recognized.")


def parse_edges(state):
    if isinstance(state, str):
        return edge_string_to_edge_tuples(state)
    if isinstance(state, list):
        if not all(isinstance(item, tuple) for item in state):
            raise Exception(f"Input format for {state} not recognized.")
        new_state = []
        for item_1, item_2 in state:
            if isinstance(item_1, str) or isinstance(item_1, int):
                item_1 = str(item_1)
            if isinstance(item_2, str) or isinstance(item_2, int):
                item_2 = str(item_2)
            if not isinstance(item_1, str) or not isinstance(item_2, str):
                raise Exception(f"Input format for {state} not recognized.")
            new_state.append((item_1, item_2))
        return new_state
    if isinstance(state, tuple):
        return [state]
    if isinstance(state, nx.DiGraph):
        new_state = []
        for edge in state.edges():
            left, right = edge
            if isinstance(left, str) or isinstance(left, int):
                left = str(left)
            if isinstance(right, str) or isinstance(right, int):
                right = str(right)
            if not isinstance(left, str) or not isinstance(right, str):
                raise Exception(f"Input format for {state} not recognized.")
            new_state.append((left, right))
        return new_state
    raise Exception(f"Input format for {state} not recognized: {type(state)}")


def parse_tuples_str_int_list(state):
    if isinstance(state, list):
        if all(isinstance(item, tuple) for item in state):
            return [parse_tuple_str_int(item) for item in state]
    if isinstance(state, tuple):
        return [parse_tuple_str_int(state)]
    raise Exception(f"Input format for {state} not recognized.")


def parse_tuple_str_int(state):
    if isinstance(state, tuple):
        item_1, item_2 = state
        if isinstance(item_1, str) or isinstance(item_1, int):
            item_1 = str(item_1)
        if isinstance(item_2, str) or isinstance(item_2, int):
            item_2 = int(item_2)
        if not isinstance(item_1, str) or not isinstance(item_2, int):
            raise Exception(f"Tuple input format for {state} not recognized.")
        return (item_1, item_2)
    raise Exception(f"Input format for {state} not recognized: {type(state)}")


def edge_string_to_edge_tuples(edges: str) -> list[tuple]:
    edge_tuples = []
    edges_part = edges.split(",")

    for part in edges_part:
        part = part.strip()
        left, right = part.split("->")
        left = left.strip()
        right = right.strip()
        edge_tuples.append((left, right))
    return edge_tuples


def list_tuples_into_list_nodes(
    list_tuples_label_value: list[tuple[str, int]], graph: Graph
) -> list[Node] | None:
    if list_tuples_label_value is None or len(list_tuples_label_value) <= 0:
        return None
    if len(list_tuples_label_value) == 1:
        return [tuple_into_node(list_tuples_label_value[0], graph)]
    return [tuple_into_node(tuple, graph) for tuple in list_tuples_label_value]


def tuple_into_node(tuple_label_value: tuple[str, int], graph: Graph) -> Node | None:
    if tuple_label_value is None:
        return None
    label, value = tuple_label_value
    if not graph.is_node_in_graph(label):
        raise Exception(f"Node '{label}' not present in the defined graph.")
    graph.set_node_value(label, value)
    return graph.graphNodes[label]


def parse_input_graph(
    edges: list[tuple[str, str]],
    latents_label: list[str],
    custom_cardinalities: dict[str, int],
):
    return parse_default_graph(edges, latents_label, custom_cardinalities)


def parse_default_graph(
    edge_tuples: list[tuple],
    latents_label: list[str],
    custom_cardinalities: dict[str, int] = {},
) -> tuple[
    int,
    dict[str, list[str]],
    dict[str, int],
    dict[str, list[str]],
    set[str],
    nx.DiGraph,
]:
    node_labels_set = set()
    children: dict[str, list[str]] = {}
    parents: dict[str, list[str]] = {}
    dag: nx.DiGraph = nx.DiGraph()

    for each_tuple in edge_tuples:
        left, right = each_tuple
        if right in latents_label:
            raise Exception(f"Invalid latent node: {right}. Latent has income arrows.")

        node_labels_set.add(left)
        node_labels_set.add(right)

        children.setdefault(left, []).append(right)
        parents.setdefault(left, [])

        parents.setdefault(right, []).append(left)
        children.setdefault(right, [])

        dag.add_edge(left, right)

    for node_label in latents_label:
        if node_label not in node_labels_set:
            raise Exception(
                f"Invalid latent node: {node_label}. Not present in the graph."
            )

    number_of_nodes = len(node_labels_set)

    node_cardinalities: dict[str, int] = {}
    for node_label in node_labels_set:
        if node_label in custom_cardinalities:
            node_cardinalities[node_label] = custom_cardinalities[node_label]
        else:
            node_cardinalities[node_label] = 0 if node_label in latents_label else 2
    return number_of_nodes, children, node_cardinalities, parents, node_labels_set, dag
