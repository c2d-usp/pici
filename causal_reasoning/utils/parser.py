import logging
from typing import TypeVar

import networkx as nx

from causal_reasoning.graph.graph import Graph
from causal_reasoning.graph.node import Node

logger = logging.getLogger(__name__)

T = TypeVar("str")


class Parser:
    def __init__(
        self,
        edges: T,
        custom_cardinalities: dict[T, int] | None = {},
        unobservables_labels: list[T] | T | None = [],
        interventions: list[tuple[T, int]] | tuple[T, int] = [],
        target: tuple[T, int] = None,
    ) -> None:
        edges = _parse_edges(edges)

        unobservables_labels = _parse_to_string_list(unobservables_labels)

        self.graph: Graph = _define_graph(
            edges=edges,
            unobservables=unobservables_labels,
            custom_cardinalities=custom_cardinalities,
        )

        self.unobservables = [
            self.graph.graphNodes[unobservable_label]
            for unobservable_label in unobservables_labels
        ]

        if interventions:
            interventions = convert_tuples_list_into_nodes_list(
                _parse_tuples_str_int_list(interventions), self.graph
            )
        self.interventions = interventions

        if target:
            target = convert_tuple_into_node(_parse_tuple_str_int(target), self.graph)
        self.target = target

    def get_graph(self) -> Graph:
        return self.graph

    def get_interventions(self) -> list[Node]:
        return self.interventions

    def get_target(self) -> Node:
        return self.target

    def get_unobservables(self) -> list[Node]:
        return self.unobservables


def _parse_edges(state):
    if isinstance(state, str):
        return _convert_edge_string_to_edge_tuples(state)
    elif isinstance(state, nx.DiGraph):
        output = []
        for left, right in state.edges():
            output.append(_pair_to_valid_tuple(left, right))
        return output
    elif isinstance(state, tuple):
        if len(state) != 2:
            raise ValueError(
                f"Input format for {state} not recognized (tuple must be length 2)."
            )
        return [_pair_to_valid_tuple(state[0], state[1])]
    elif isinstance(state, list):
        if not all(isinstance(item, tuple) and len(item) == 2 for item in state):
            raise ValueError(
                f"Input format for {state} not recognized (list must contain 2‐tuples)."
            )
        output = []
        for item_1, item_2 in state:
            output.append(_pair_to_valid_tuple(item_1, item_2))
        return output
    else:
        raise ValueError(f"Input format for {state} not recognized: {type(state)}")


def _convert_edge_string_to_edge_tuples(edges: str) -> list[tuple]:
    edge_tuples = []
    edges_part = edges.split(",")

    for part in edges_part:
        part = part.strip()
        left, right = part.split("->")
        left = left.strip()
        right = right.strip()
        edge_tuples.append((left, right))
    return edge_tuples


def _pair_to_valid_tuple(left, right):
    if isinstance(left, (str, int)):
        left = str(left)
    if isinstance(right, (str, int)):
        right = str(right)
    if not isinstance(left, str) or not isinstance(right, str):
        raise ValueError(f"Input format for ({left}, {right}) not recognized.")
    return (left, right)


def convert_tuples_list_into_nodes_list(
    list_tuples_label_value: list[tuple[str, int]], graph: Graph
) -> list[Node] | None:
    if not list_tuples_label_value:
        return None

    output = []
    for item in list_tuples_label_value:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(f"Expected list of 2-tuples, got {item!r}")
        output.append(convert_tuple_into_node(item, graph))
    return output


def convert_tuple_into_node(
    tuple_label_value: tuple[str, int], graph: Graph
) -> Node | None:
    if tuple_label_value is None:
        return None
    label, value = tuple_label_value
    if not graph.is_node_in_graph(label):
        raise Exception(f"Node '{label}' not present in the defined graph.")
    graph.set_node_intervened_value(label, value)
    return graph.graphNodes[label]


def _parse_tuples_str_int_list(state):
    if isinstance(state, list):
        if all(isinstance(item, tuple) for item in state):
            return [_parse_tuple_str_int(item) for item in state]
    if isinstance(state, tuple):
        return [_parse_tuple_str_int(state)]
    raise Exception(f"Input format for {state} not recognized.")


def _parse_tuple_str_int(state):
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


def _parse_to_string_list(state):
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


def _define_graph(
    edges: tuple[str, str] = None,
    unobservables: list[str] = None,
    custom_cardinalities: dict[str, int] = {},
):
    (
        number_of_nodes,
        children_labels,
        node_cardinalities,
        parents_labels,
        node_labels_set,
        dag,
    ) = _parse_input_graph(
        edges, latents_label=unobservables, custom_cardinalities=custom_cardinalities
    )
    order = list(nx.topological_sort(dag))

    parent_latent_labels: dict[str, str] = {}
    graphNodes: dict[str, Node] = {}
    node_set: set[Node] = set()

    parent_latent_label: str = None
    for node_label in node_labels_set:
        if node_cardinalities[node_label] == 0:
            parent_latent_label = None
            new_node = Node(
                children=[],
                parents=[],
                latentParent=None,
                isLatent=True,
                label=node_label,
                cardinality=node_cardinalities[node_label],
            )
        else:
            parent_latent_label = _get_parent_latent(
                parents_labels[node_label], node_cardinalities
            )

            if parent_latent_label is None:
                logger.error(
                    f"PARSE ERROR: ALL OBSERVABLE VARIABLES SHOULD HAVE A LATENT PARENT, BUT {node_label} DOES NOT."
                )

            new_node = Node(
                children=[],
                parents=[],
                latentParent=None,
                isLatent=False,
                label=node_label,
                cardinality=node_cardinalities[node_label],
            )

        graphNodes[node_label] = new_node
        parent_latent_labels[new_node.label] = parent_latent_label
        node_set.add(new_node)

    endogenous: list[Node] = []
    exogenous: list[Node] = []
    topologicalOrderIndexes = {}

    for i, node_label in enumerate(node_labels_set):
        node = graphNodes[node_label]
        if node.isLatent:
            exogenous.append(node)
            node.children = _get_node_list(graphNodes, children_labels[node.label])
        else:
            node.latentParent = graphNodes[parent_latent_labels[node_label]]
            endogenous.append(node)
            node.children = _get_node_list(graphNodes, children_labels[node.label])
            node.parents = _get_node_list(graphNodes, parents_labels[node.label])
        topologicalOrderIndexes[node] = i

    topological_order_nodes: list[Node] = []
    for node_label in order:
        topological_order_nodes.append(graphNodes[node_label])

    return Graph(
        numberOfNodes=number_of_nodes,
        exogenous=exogenous,  # list[Node]
        endogenous=endogenous,  # list[Node]
        topologicalOrder=topological_order_nodes,  # list[Node]
        DAG=dag,  # nx.DiGraph
        graphNodes=graphNodes,  # dict[str, Node]
        node_set=node_set,  # set(Node)
        topologicalOrderIndexes=topologicalOrderIndexes,  # dict[Node, int]
        currNodes=[],
        dagComponents=[],  # list[list[Node]]
        cComponentToUnob={},  # dict[int, Node]
    )


def _parse_input_graph(
    edges: list[tuple[str, str]],
    latents_label: list[str],
    custom_cardinalities: dict[str, int],
):
    return _parse_default_graph(edges, latents_label, custom_cardinalities)


def _parse_default_graph(
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


def _get_parent_latent(parents_label: list[str], node_cardinalities: list[str]) -> str:
    for node_parent in parents_label:
        if node_cardinalities[node_parent] == 0:
            return node_parent
    return None


def _get_node_list(graphNodes: dict[str, Node], node_labels: list[str]) -> list[Node]:
    return [_get_node(graphNodes, node_label) for node_label in node_labels]


def _get_node(graphNodes: dict[str, Node], node_label: str):
    return graphNodes[node_label]
