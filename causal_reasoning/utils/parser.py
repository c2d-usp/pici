import networkx as nx
from typing import Any


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


def parse_input_graph(edges: Any, latents: list[str], custom_cardinalities: dict[str, int]={}):
    edge_tuples = parse_edges(edges)
    return parse_default_input(edge_tuples, latents, custom_cardinalities)


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
    edge_tuples: list[tuple], latents: list[str], custom_cardinalities: dict[str, int]):    
    node_set = set()
    adjacency_list: dict[str, list[str]] = {}
    parents: dict[str, list[str]] = {}
    dag: nx.DiGraph = nx.DiGraph()

    for each_tuple in edge_tuples:
        left, right = each_tuple
        if right in latents:
            raise Exception(f"Invalid latent node: {right}. Latent has income arrows.")
        
        node_set.add(left)
        node_set.add(right)

        if left not in adjacency_list:
            adjacency_list[left] = []
        adjacency_list[left].append(right)

        # APENAS PARA DEIXAR VAZIO
        if left not in parents:
            parents[left] = []

        if right not in parents:
            parents[right] = []    
        parents[right].append(left)

        # APENAS PARA DEIXAR VAZIO
        if right not in adjacency_list:
            adjacency_list[right] = []

        dag.add_edge(left, right)

    for node in latents:
        if node not in node_set:
            raise Exception(f"Invalid latent node: {node}. Not present in the graph.")
    
    number_of_nodes = len(node_set)

    node_cardinalities: dict[str, int] = {}
    for node in node_set:
        if node in custom_cardinalities:
            node_cardinalities[node] = custom_cardinalities[node]
        else:
            node_cardinalities[node] = 0 if node in latents else 2
    return number_of_nodes, adjacency_list, node_cardinalities, parents, node_set, dag
