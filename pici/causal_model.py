import copy
import logging
from typing import Any, TypeVar

import networkx as nx
from pandas import DataFrame

logger = logging.getLogger(__name__)
logging.getLogger("pgmpy").setLevel(logging.WARNING)
logging.getLogger("dowhy.causal_model").setLevel(logging.ERROR)

from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference.CausalInference import CausalInference
from pgmpy.models import DiscreteBayesianNetwork

from pici.graph.graph import Graph
from pici.graph.node import Node
from pici.intervention_inference_algorithm.linear_programming.opt_problem_builder import (
    build_linear_problem,
)
from pici.utils._enum import OptimizersLabels
from pici.utils.graph_plotter import plot_graph_image
from pici.utils.parser import (
    Parser,
    convert_tuple_into_node,
    convert_tuples_list_into_nodes_list,
)

from .identifier import Identifier

T = TypeVar("str")


class CausalModel:
    def __init__(
        self,
        data: DataFrame,
        edges: T,
        unobservables_labels: list[T] | T,
        custom_cardinalities: dict[T, int] | None = {},
        interventions: list[tuple[T, int]] | tuple[T, int] = [],
        target: tuple[T, int] = None,
    ) -> None:
        self.data = data

        parser = Parser(
            edges, unobservables_labels, custom_cardinalities, interventions, target
        )

        self.graph: Graph = parser.get_graph()
        self.unobservables: list[Node] = parser.get_unobservables()
        self.interventions: list[Node] = parser.get_interventions()
        self.target: Node = parser.get_target()

        del parser

    def intervention_query(
        self,
        interventions: list[tuple[str, int]] = None,
        target: tuple[str, int] = None,
    ) -> str | tuple[str, str]:
        """
        Return the intervention query, for identifiable and partially identifiable cases.

        Args:
            interventions: A list of tuples representing the interventions.
            target: A tuple representing the target variable.

        Returns:
            The intervention query, as either a point or a set of bounds.
        """
        is_identifiable, _, _ = self.is_identifiable_intervention(
            interventions=interventions, target=target
        )
        if is_identifiable:
            return self.identifiable_intervention_query(
                interventions=interventions, target=target
            )
        return self.partially_identifiable_intervention_query(
            interventions=interventions, target=target
        )

    def is_identifiable_intervention(
        self,
        interventions: list[tuple[str, int]] = None,
        target: tuple[str, int] = None,
    ) -> tuple[bool, str | None, Any | None]:
        """
        Check if the intervention is identifiable.

        Args:
            interventions: A list of tuples representing the interventions.
            target: A tuple representing the target variable.

        Returns:
            A tuple containing a boolean indicating if the intervention is identifiable,
            the identification method used (if any), and the result of the identification, such as an adjustment set (if applicable).
        """

        if not self.interventions_validator(interventions):
            raise ValueError("Invalid interventions")
        if not self.target_validator(target):
            raise ValueError("Invalid target")

        identifier = Identifier(causal_model=self)

        for method in ["backdoor", "frontdoor"]:
            finder = getattr(identifier, f"find_{method}")
            result = finder()
            if result:
                return True, method, result

        if identifier.check_unobservable_confounding():
            return False, "unobservable_confounding", None

        if identifier.graphical_identification():
            return True, "graphical", None

        estimand = identifier.id_algorithm_identification()
        return (True, "id_algorithm", estimand) if estimand else (False, None, None)

    def identifiable_intervention_query(
        self,
        interventions: list[tuple[str, int]] = None,
        target: tuple[str, int] = None,
    ) -> str:
        """
        Calculate the intervention query for completely identifiable cases.

        Args:
            interventions: A list of tuples representing the interventions.
            target: A tuple representing the target variable.

        Returns:
            The point value result for the identifiable intervention query as a string.
        """
        if not self.interventions_validator(interventions):
            raise ValueError("Invalid interventions")
        if not self.target_validator(target):
            raise ValueError("Invalid target")

        G = DiscreteBayesianNetwork()
        G.add_edges_from(self.graph.DAG.edges())
        G.fit(self.data, estimator=MaximumLikelihoodEstimator)
        model = CausalInference(G)
        min_adjustment_set = model.get_minimal_adjustment_set(
            X=self.interventions[0].label, Y=self.target.label
        )

        distribution = model.query(
            variables=[self.target.label],
            do={
                self.interventions[i].label: self.interventions[i].intervened_value
                for i in range(len(self.interventions))
            },
            adjustment_set=min_adjustment_set,
        )

        kwargs = {}
        kwargs[self.target.label] = self.target.intervened_value

        return distribution.get_value(**kwargs)

    def partially_identifiable_intervention_query(
        self,
        interventions: list[tuple[str, int]] = None,
        target: tuple[str, int] = None,
    ) -> tuple[str, str]:
        """
        Calculate the intervention query for partially identifiable cases.

        Args:
            interventions: A list of tuples representing the interventions.
            target: A tuple representing the target variable.

        Returns:
            The bounds of the partially identifiable intervention query as a tuple of strings.
        """
        if not self.interventions_validator(interventions):
            raise ValueError("Invalid interventions")
        if not self.target_validator(target):
            raise ValueError("Invalid target")

        if len(self.interventions) == 1:
            return self.single_intervention_query()
        elif len(self.interventions) >= 2:
            self.multi_intervention_query()
            return ("None", "None")
        raise Exception("None interventions found. Expect at least one intervention.")

    def single_intervention_query(self) -> tuple[str, str]:
        """
        Calculate the intervention query for a single intervention partially identifiable case.
        """
        return build_linear_problem(
            graph=self.graph,
            df=self.data,
            intervention=self.interventions[0],
            target=self.target,
            optimizer_label=OptimizersLabels.GUROBI.value,
        )

    def multi_intervention_query(self):
        raise NotImplementedError

    def interventions_validator(
        self, interventions: list[tuple[str, int]] = None
    ) -> bool:
        """
        Validate that interventions is a non-empty list of (label, value) tuples
        and that each label is actually in the graph.
        """
        if interventions is None:
            if not getattr(self, "interventions", None):
                raise ValueError("Expected a non-empty list of interventions")
            return True

        if not interventions:
            raise ValueError("Expected a non-empty list of interventions")

        nodes = convert_tuples_list_into_nodes_list(interventions, self.graph)
        if nodes is None:
            raise ValueError("Could not map interventions to graph nodes")

        dag_nodes = set(self.graph.DAG.nodes())
        for node in nodes:
            if node.label not in dag_nodes:
                raise ValueError(f"Intervention '{node.label}' is not in the graph")

        self.interventions = nodes
        return True

    def target_validator(self, target: tuple[str, int] = None) -> bool:
        """
        Validate that target is a (label, value) tuple whose label is
        actually in the graph.
        """
        if target is None:
            if not getattr(self, "target", None):
                raise ValueError("Expected a target (node_label, value) tuple")
            return True

        node = convert_tuple_into_node(target, self.graph)
        if node is None:
            raise ValueError("Could not map target to a graph node")

        dag_nodes = set(self.graph.DAG.nodes())
        if node.label not in dag_nodes:
            raise ValueError(f"Target '{node.label}' is not in the graph")

        self.target = node
        return True

    def weak_pn_inference(
        self, intervention_label: str, target_label: str
    ) -> str | tuple[str, str]:
        """
        Calculate the weak PN intervention query.

        PN = P(Y_{X=0} = 0 | X = 1, Y = 1)
        WEAK_PN = P(Y_{X=0} = 0)

        Args:
            intervention_label: The label of the intervention variable.
            target_label: The label of the target variable.

        Returns:
            The weak PN intervention query point value result as a string, if identifiable, or bounds as a tuple of strings, if partially identifiable.
        """
        return self.intervention_query(
            interventions=[(intervention_label, 0)], target=(target_label, 0)
        )

    def weak_ps_inference(
        self, intervention_label: str, target_label: str
    ) -> str | tuple[str, str]:
        """
        Calculate the weak PS intervention query.

        PS = P(Y_{X=1} = 1 | X = 0, Y = 0)
        WEAK_PS = P(Y_{X=1} = 1)

        Args:
            intervention_label: The label of the intervention variable.
            target_label: The label of the target variable.

        Returns:
            The weak PS intervention query point value result as a string, if identifiable, or bounds as a tuple of strings, if partially identifiable.
        """
        return self.intervention_query(
            interventions=[(intervention_label, 1)], target=(target_label, 1)
        )

    def are_d_separated_in_intervened_graph(
        self,
        set_nodes_X: list[str],
        set_nodes_Y: list[str],
        set_nodes_Z: list[str],
        G: nx.DiGraph = None,
    ) -> bool:
        """
        Check if set of nodes X is d-separated from set of nodes Y through set of nodes Z in the intervened graph.

        Args:
            set_nodes_X: The set of nodes X.
            set_nodes_Y: The set of nodes Y.
            set_nodes_Z: The set of nodes Z.
            G: The graph to check d-separation in.

        Returns:
            True if X is d-separated from Y given Z in the intervened graph, False otherwise.

        """
        if G is None:
            G = self.graph.DAG

        if len(self.interventions) <= 0:
            return self.are_d_separated_in_complete_graph(
                set_nodes_X, set_nodes_Y, set_nodes_Z, G
            )

        operated_digraph = copy.deepcopy(G)
        interventions_outgoing_edges = []
        for intervention in self.interventions:
            interventions_outgoing_edges.extend(list(G.in_edges(intervention.label)))
        operated_digraph.remove_edges_from(interventions_outgoing_edges)

        return nx.is_d_separator(
            G=operated_digraph,
            x=set(set_nodes_X),
            y=set(set_nodes_Y),
            z=set(set_nodes_Z),
        )

    def are_d_separated_in_complete_graph(
        self,
        set_nodes_X: list[str],
        set_nodes_Y: list[str],
        set_nodes_Z: list[str],
        G: nx.DiGraph = None,
    ) -> bool:
        """
        Check if set of nodes X is d-separated from set of nodes Y through set of nodes Z in the complete graph.

        Args:
            set_nodes_X: The set of nodes X.
            set_nodes_Y: The set of nodes Y.
            set_nodes_Z: The set of nodes Z.
            G: The graph to check d-separation in.

        Returns:
            True if X is d-separated from Y given Z in the complete graph, False otherwise.
        """
        if G is None:
            G = self.graph.DAG
        return nx.is_d_separator(
            G, set(set_nodes_X), set(set_nodes_Y), set(set_nodes_Z)
        )

    def set_interventions(self, interventions: list[tuple[str, int]]) -> None:
        """
        Set the interventions for the causal model.

        Args:
            interventions: A list of tuples representing the interventions.
        """
        self.interventions = convert_tuples_list_into_nodes_list(
            interventions, self.graph
        )

    def add_interventions(self, interventions: list[tuple[str, int]]) -> None:
        """
        Add interventions to the causal model.

        Args:
            interventions: A list of tuples representing the interventions.
        """
        more_interventions = convert_tuples_list_into_nodes_list(
            interventions, self.graph
        )
        if more_interventions is None:
            return
        for intervention in more_interventions:
            if intervention not in self.interventions:
                self.interventions.append(intervention)

    def set_target(self, target: tuple[str, int]) -> None:
        """
        Set the target for the causal model.

        Args:
            target: A tuple representing the target.
        """
        self.target = convert_tuple_into_node(target, self.graph)

    def generate_graph_image(self, output_path="graph.png"):
        """
        Generate and save an image of the graph, using networkx.

        Args:
            output_path: The path to save the graph image.
        """
        plot_graph_image(
            graph=self.graph.DAG,
            unobservables=self.unobservables,
            interventions=self.interventions,
            targets=[self.target],
            output_path=output_path,
        )
