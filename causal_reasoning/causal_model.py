import copy
from typing import TypeVar
import networkx as nx
from pandas import DataFrame
import logging

logger = logging.getLogger(__name__)

from causal_reasoning.graph.graph import Graph
from causal_reasoning.graph.node import Node
from causal_reasoning.interventional_do_calculus_algorithm.gurobi_use import gurobi_build_linear_problem
from causal_reasoning.interventional_do_calculus_algorithm.opt_problem_builder import (
    build_bi_linear_problem,
    build_linear_problem,
)
from causal_reasoning.utils.parser import (
    list_tuples_into_list_nodes,
    parse_edges,
    parse_input_graph,
    parse_to_string_list,
    parse_tuples_str_int_list,
    parse_tuple_str_int,
    tuple_into_node,
)

T = TypeVar("str")


class CausalModel:
    def __init__(
        self,
        data: DataFrame,
        edges: T,
        custom_cardinalities: dict[T, int] | None = {},
        unobservables_labels: list[T] | T | None = [],
        interventions: list[tuple[T, int]] | tuple[T, int] = [],
        target: tuple[T, int] = None,
    ) -> None:
        self.data = data

        # TODO: If it is a nx
        # Or str
        # or list of tuples
        # or one tuple
        edges = parse_edges(edges)

        unobservables_labels = parse_to_string_list(unobservables_labels)
        self.graph: Graph = get_graph(
            edges=edges,
            unobservables=unobservables_labels,
            custom_cardinalities=custom_cardinalities,
        )
        self.unobservables = [
            self.graph.graphNodes[unobservable_label]
            for unobservable_label in unobservables_labels
        ]
        # TODO:
        if interventions:
            interventions = list_tuples_into_list_nodes(
                parse_tuples_str_int_list(interventions), self.graph
            )
        self.interventions = interventions

        if target:
            target = tuple_into_node(parse_tuple_str_int(target), self.graph)
        self.target = target

    def are_d_separated_in_complete_graph(
        self,
        set_nodes_X: list[str],
        set_nodes_Y: list[str],
        set_nodes_Z: list[str],
        G: nx.DiGraph = None,
    ) -> bool:
        """
        Is set of nodes X d-separated from set of nodes Y through set of nodes Z?

        Given two sets of nodes (nodes1 and nodes2), the function returns true if every node in nodes1
        is independent of every node in nodes2, given that the nodes in conditionedNodes are conditioned.
        """
        if G is None:
            G = self.graph.DAG
        return nx.is_d_separator(G, set(set_nodes_X), set(set_nodes_Y), set(set_nodes_Z))

    def are_d_separated_in_intervened_graph(
        self,
        set_nodes_X: list[str],
        set_nodes_Y: list[str],
        set_nodes_Z: list[str],
        G: nx.DiGraph = None,
    ) -> bool:
        if G is None:
            G = self.graph.DAG

        if len(self.interventions) <= 0:
            return self.are_d_separated_in_complete_graph(set_nodes_X,set_nodes_Y,set_nodes_Z,G)

        operated_digraph = copy.deepcopy(G)
        interventions_outgoing_edges = []
        for intervention in self.interventions:
            interventions_outgoing_edges.extend(list(G.out_edges(intervention.label)))
        operated_digraph.remove_edges_from(interventions_outgoing_edges)
        return nx.is_d_separator(G=operated_digraph, x=set(set_nodes_X), y=set(set_nodes_Y), z=set(set_nodes_Z))

    def inference_intervention_query(
        self, interventions: list[tuple[str, int]] = [], target: tuple[str, int] = None
    ) -> tuple[str, str]:
        interventions_nodes = list_tuples_into_list_nodes(interventions, self.graph)
        if interventions_nodes is None and self.interventions is None:
            raise Exception("Expect intervention to be not None")

        if interventions_nodes is not None:
            self.interventions = interventions_nodes

        target_node = tuple_into_node(target, self.graph)
        if target_node is None and self.target is None:
            raise Exception("Expect target to be not None")
        if target_node is not None:
            self.target = target_node

        if len(self.interventions) == 1:
            return self.single_intervention_query()
        elif len(self.interventions) == 2:
            return self.double_intervention_query()
        elif len(self.interventions) > 2:
            self.multi_intervention_query()
            return ("None", "None")
        raise Exception("None interventions found")

    def single_intervention_query(self) -> tuple[str, str]:
        # return build_linear_problem(
        return gurobi_build_linear_problem(
            self.graph,
            self.data,
            self.interventions[0],
            self.target,
        )

    def double_intervention_query(self):
        return build_bi_linear_problem(
            self.graph,
            self.data,
            self.interventions,
            self.target,
        )

    def multi_intervention_query(self):
        raise NotImplementedError

    def set_interventions(self, interventions: list[tuple[str, int]]) -> None:
        self.interventions = list_tuples_into_list_nodes(interventions, self.graph)

    def add_interventions(self, interventions: list[tuple[str, int]]) -> None:
        more_interventions = list_tuples_into_list_nodes(interventions, self.graph)
        if more_interventions is None:
            return
        for intervention in more_interventions:
            if intervention not in self.interventions:
                self.interventions.append(intervention)

    def set_target(self, target: tuple[str, int]) -> None:
        self.target = tuple_into_node(target, self.graph)

    def set_unobservables(self, unobservables):
        # This implies the whole graph re-creation
        # Changes the intervention and target also (?)
        raise NotImplementedError

    def add_unobservables(self, unobservables):
        # This implies the whole graph re-creation
        # Changes the intervention and target also (?)
        raise NotImplementedError

    def visualize_graph(self):
        raise NotImplementedError


def get_node(graphNodes: dict[str, Node], node_label: str):
    return graphNodes[node_label]


def get_node_list(graphNodes: dict[str, Node], node_labels: list[str]) -> list[Node]:
    return [get_node(graphNodes, node_label) for node_label in node_labels]


def get_parent_latent(parents_label: list[str], node_cardinalities: list[str]) -> str:
    for node_parent in parents_label:
        if node_cardinalities[node_parent] == 0:
            return node_parent
    return None


def get_graph(
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
    ) = parse_input_graph(
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
            parent_latent_label = get_parent_latent(
                parents_labels[node_label], node_cardinalities
            )

            if parent_latent_label is None:
                # TODO: ADD WARNING MESSAGE
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
            node.children = get_node_list(graphNodes, children_labels[node.label])
        else:
            node.latentParent = graphNodes[parent_latent_labels[node_label]]
            endogenous.append(node)
            node.children = get_node_list(graphNodes, children_labels[node.label])
            node.parents = get_node_list(graphNodes, parents_labels[node.label])
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
