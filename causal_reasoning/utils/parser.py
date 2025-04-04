import networkx as nx
from typing import Any

from causal_reasoning.graph.node import T


def parse_to_string_list(state):
    if isinstance(state, str):
        return [state]
    if isinstance(state, list):
        if all(isinstance(item, str) for item in state):
            return state
    raise Exception(f"Input format for {state} not recognized.")


def parse_to_string(state):
    if isinstance(state, str):
        return state
    raise Exception(f"Input format for {state} not recognized: {type(state)}")


def parse_to_int(state):
    if isinstance(state, int):
        return state
    raise Exception(f"Input format for {state} not recognized: {type(state)}")


def parse_to_int_list(state):
    if isinstance(state, int):
        return [state]
    if isinstance(state, list):
        if all(isinstance(item, int) for item in state):
            return state
    raise Exception(f"Input format for {state} not recognized.")


def parse_interventions(intervention_variables, intervention_values):
    if isinstance(intervention_variables, str) and isinstance(intervention_values, int):
        return


def parse_input_graph(
    edges: Any, latents_label: list[T], custom_cardinalities: dict[T, int] = {}
):
    edge_tuples = parse_edges(edges)
    return parse_default_input(edge_tuples, latents_label, custom_cardinalities)


def parse_edges(state):
    if isinstance(state, str):
        return edge_string_to_edge_tuples(state)
    if isinstance(state, list):
        # TODO: Validate if the Tuple is (str, str)
        # TODO: Also consider int variables (int, int)
        if all(isinstance(item, tuple) for item in state):
            return state
        raise Exception(f"Input format for {state} not recognized.")
    if isinstance(state, tuple):
        return [state]
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


def parse_default_input(
    edge_tuples: list[tuple], latents_label: list[T], custom_cardinalities: dict[T, int]
) -> tuple[int, dict[T, list[T]], dict[T, int], dict[T, list[T]], set[T], nx.DiGraph]:
    node_labels_set = set()
    children: dict[T, list[T]] = {}
    parents: dict[T, list[T]] = {}
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

    node_cardinalities: dict[T, int] = {}
    for node_label in node_labels_set:
        if node_label in custom_cardinalities:
            node_cardinalities[node_label] = custom_cardinalities[node_label]
        else:
            node_cardinalities[node_label] = 0 if node_label in latents_label else 2
    return number_of_nodes, children, node_cardinalities, parents, node_labels_set, dag
