import copy
import logging
import warnings
from typing import TypeVar

import networkx as nx
from pandas import DataFrame
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference.CausalInference import CausalInference
from pgmpy.estimators import MaximumLikelihoodEstimator
from dowhy import CausalModel as DowhyCausalModel

from causal_reasoning.utils.graph_plotter import plot_graph_image

logger = logging.getLogger(__name__)
logging.getLogger("pgmpy").setLevel(logging.WARNING)
logging.getLogger("dowhy.causal_model").setLevel(logging.ERROR)

from causal_reasoning.do_calculus_algorithm.linear_programming.opt_problem_builder import (
    build_bi_linear_problem,
    build_linear_problem,
)
from causal_reasoning.graph.graph import Graph
from causal_reasoning.graph.node import Node
from causal_reasoning.utils._enum import OptimizersLabels
from causal_reasoning.utils.parser import (
    convert_tuples_list_into_nodes_list,
    convert_tuple_into_node,
    Parser,
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

        parser = Parser(
            edges, custom_cardinalities, unobservables_labels, interventions, target
        )

        self.graph: Graph = parser.get_graph()
        self.unobservables: list[Node] = parser.get_unobservables()
        self.interventions: list[Node] = parser.get_interventions()
        self.target: Node = parser.get_target()

        del parser

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
        return nx.is_d_separator(
            G, set(set_nodes_X), set(set_nodes_Y), set(set_nodes_Z)
        )

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
    
    def is_identifiable_intervention(
        self, interventions: list[tuple[str, int]] = [], target: tuple[str, int] = None
    ) -> bool:
        """
        Check if the intervention is identifiable.
        """
        interventions_nodes = convert_tuples_list_into_nodes_list(interventions, self.graph)
        if interventions_nodes is None and self.interventions is None:
            raise Exception("Expect intervention to be not None")

        if interventions_nodes is not None:
            self.interventions = interventions_nodes

        target_node = convert_tuple_into_node(target, self.graph)
        if target_node is None and self.target is None:
            raise Exception("Expect target to be not None")
        if target_node is not None:
            self.target = target_node

        X = self.interventions[0].label
        Y = self.target.label
        latent_labels = {u.label for u in (self.unobservables or [])}

        dbn = DiscreteBayesianNetwork()
        dbn.add_edges_from(self.graph.DAG.edges())

        infer = CausalInference(dbn)

        backdoors = infer.get_all_backdoor_adjustment_sets(X=X, Y=Y)
        obs_bds = [Z for Z in backdoors if Z.isdisjoint(latent_labels)]
        if obs_bds:
            Z = min(obs_bds, key=len)
            return True, "backdoor", Z
        
        frontdoors = infer.get_all_frontdoor_adjustment_sets(X=X, Y=Y)
        obs_fds = [Z for Z in frontdoors if Z.isdisjoint(latent_labels)]
        if obs_fds:
            W = min(obs_fds, key=len)
            return True, "frontdoor", W

        def all_directed_paths(G, src, dst):
            for path in nx.all_simple_paths(G, src, dst):
                if all(G.has_edge(u, v) for u, v in zip(path, path[1:])):
                    yield path

        G = self.graph.DAG
        observed = set(G.nodes()) - latent_labels
        for z in observed - {X, Y}:
            if not G.has_edge(z, X):
                continue
            bds_zy = infer.get_all_backdoor_adjustment_sets(X=z, Y=Y)
            if any(not Bd.isdisjoint(latent_labels) for Bd in bds_zy):
                continue
            if all(X in path[1:-1] for path in all_directed_paths(G, z, Y)):
                return True, "iv", z

        data_obs = self.data.drop(
            columns=[u for u in latent_labels if u in self.data.columns]
        )

        for U in (self.unobservables or []):
            if nx.has_path(G, U.label, X) and nx.has_path(G, U.label, Y):
                return False, None, None
            
        G_do = G.copy()
        G_do.remove_edges_from(list(G_do.in_edges(X)))
        if nx.is_d_separator(G_do, {X}, {Y}, set()):
            return True, "graphical", None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dowhy_model = DowhyCausalModel(
                data=data_obs,
                graph=self.graph.DAG,
                treatment=X,
                outcome=Y
            )
            try:
                estimand = dowhy_model.identify_effect(
                    method_name="id-algorithm",
                    proceed_when_unidentifiable=False
                )
                return True, "id-algorithm", estimand
            except Exception:
                return False, None, None

    def identifiable_intervention_query(
        self, interventions: list[tuple[str, int]] = [], target: tuple[str, int] = None
    ) -> str:
        interventions_nodes = convert_tuples_list_into_nodes_list(
            interventions, self.graph
        )
        if interventions_nodes is None and self.interventions is None:
            raise Exception("Expect intervention to be not None")

        if interventions_nodes is not None:
            self.interventions = interventions_nodes

        target_node = convert_tuple_into_node(target, self.graph)
        if target_node is None and self.target is None:
            raise Exception("Expect target to be not None")
        if target_node is not None:
            self.target = target_node

        G = DiscreteBayesianNetwork()
        G.add_edges_from(self.graph.DAG.edges())
        G.fit(self.data, estimator=MaximumLikelihoodEstimator)
        model = CausalInference(G)
        if not self.interventions or len(self.interventions) == 0:
            raise Exception("Expect interventions to contain at least one element")
        min_adjustment_set = model.get_minimal_adjustment_set(
            X=self.interventions[0].label, Y=self.target.label
        )

        distribution = model.query(
            variables=[self.target.label],
            do={
                self.interventions[i].label: self.interventions[i].value
                for i in range(len(self.interventions))
            },
            adjustment_set=min_adjustment_set,
        )

        kwargs = {}
        kwargs[self.target.label] = self.target.value

        return distribution.get_value(**kwargs)

    def inference_intervention_query(
        self, interventions: list[tuple[str, int]] = [], target: tuple[str, int] = None
    ) -> tuple[str, str]:
        interventions_nodes = convert_tuples_list_into_nodes_list(
            interventions, self.graph
        )
        if interventions_nodes is None and self.interventions is None:
            raise Exception("Expect intervention to be not None")

        if interventions_nodes is not None:
            self.interventions = interventions_nodes

        target_node = convert_tuple_into_node(target, self.graph)
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
        raise Exception("None interventions found. Expect at least one intervention.")

    def single_intervention_query(self) -> tuple[str, str]:
        return build_linear_problem(
            graph=self.graph,
            df=self.data,
            intervention=self.interventions[0],
            target=self.target,
            optimizer_label=OptimizersLabels.GUROBI.value,
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

    def is_identifiable(self):
        # backdoor
        # frontdoor
        pass

    def set_interventions(self, interventions: list[tuple[str, int]]) -> None:
        self.interventions = convert_tuples_list_into_nodes_list(
            interventions, self.graph
        )

    def add_interventions(self, interventions: list[tuple[str, int]]) -> None:
        more_interventions = convert_tuples_list_into_nodes_list(
            interventions, self.graph
        )
        if more_interventions is None:
            return
        for intervention in more_interventions:
            if intervention not in self.interventions:
                self.interventions.append(intervention)

    def set_target(self, target: tuple[str, int]) -> None:
        self.target = convert_tuple_into_node(target, self.graph)

    def set_unobservables(self, unobservables):
        # This implies the whole graph re-creation
        # Changes the intervention and target also (?)
        raise NotImplementedError

    def add_unobservables(self, unobservables):
        # This implies the whole graph re-creation
        # Changes the intervention and target also (?)
        raise NotImplementedError

    def generate_graph_image(self, output_path="graph.png"):
        """
        Draw the graph using networkx.
        """
        plot_graph_image(
            graph=self.graph.DAG,
            unobservables=self.unobservables,
            interventions=self.interventions,
            targets=[self.target],
            output_path=output_path,
        )
