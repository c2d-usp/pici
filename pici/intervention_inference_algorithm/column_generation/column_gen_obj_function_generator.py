import copy
import logging

logger = logging.getLogger(__name__)

import networkx as nx
import pandas as pd

from pici.graph.graph import Graph
from pici.graph.node import Node


class ObjFunctionGenerator:
    """
    Given an intervention and a graph, this class finds a set of restrictions that can be used to build
    a linear objective function.
    """

    def __init__(
        self,
        graph: Graph,
        dataFrame: pd.DataFrame,
        intervention: Node,
        target: Node,
    ):
        """
        graph: an instance of the personalized class graph
        intervention: X in P(Y|do(X))
        target: Y in P(Y|do(X))
        """

        self.graph = graph
        self.intervention = intervention
        self.target = target
        self.dataFrame = dataFrame

        self.empirical_probabilities_variables: list[Node] = []
        self.mechanism_variables: list[Node] = []
        self.conditional_probabilities: dict[Node, list[Node]] = {}
        self.considered_graph_nodes: list[Node] = []

        self.setup()

    def setup(self):
        self._find_linear_good_set()

    def _find_linear_good_set(self):
        """
        Runs each step of the algorithm.
        Finds a set of variables/restrictions that linearizes the problem.
        """
        intervention: Node = self.intervention
        current_targets: list[Node] = [self.target]
        intervention_latent: Node = intervention.latent_parent

        empirical_probabilities_variables: list[Node] = []
        # If V in this array then it implies P(v) in the objective function
        # If V in this array then it implies a decision function: 1(Pa(v) => v=
        # some value)
        mechanism_variables: list[Node] = []
        # If V|A,B,C in this array then it implies P(V|A,B,C) in the objective
        # function
        conditional_probabilities: dict[Node, list[Node]] = {}
        considered_graph_nodes: list[Node] = []
        while len(current_targets) > 0:
            logger.debug("---- Current targets array:")
            for tg in current_targets:
                logger.debug(f"- {tg.label}")

            current_target = (
                self.graph.get_closest_node_from_leaf_in_the_topological_order(
                    current_targets
                )
            )
            logger.debug(f"__>>{current_target.label}<<__")
            current_targets.remove(current_target)
            considered_graph_nodes.append(current_target)
            logger.debug(f"Current target: {current_target.label}")

            if not self.graph.is_descendant(
                ancestor=self.intervention, descendant=current_target
            ):
                logger.debug("------- Case 1: Not a descendant")
                empirical_probabilities_variables.append(current_target)
            elif current_target.latent_parent == intervention_latent:
                logger.debug("------- Case 2: Mechanisms")
                mechanism_variables.append(current_target)
                for parent in current_target.parents:
                    if (parent not in current_targets) and parent != intervention:
                        current_targets.append(parent)
            else:
                logger.debug("------- Case 3: Find d-separator set")
                valid_d_separator_set = self._find_d_separator_set(
                    current_target, current_targets, intervention_latent, intervention
                )

                current_targets = list(
                    (set(current_targets) | set(valid_d_separator_set))
                    - {current_target}
                )

                conditional_probabilities[current_target] = valid_d_separator_set

        self.empirical_probabilities_variables = empirical_probabilities_variables
        self.mechanism_variables = mechanism_variables
        self.conditional_probabilities = conditional_probabilities
        self.considered_graph_nodes = considered_graph_nodes
        logger.debug("Test completed")

    def _find_d_separator_set(
        self,
        current_target: Node,
        current_targets: list[Node],
        intervention_latent: Node,
        intervention: Node,
    ):
        always_conditioned_nodes: list[Node] = current_targets.copy()
        if current_target in always_conditioned_nodes:
            always_conditioned_nodes.remove(current_target)

        if intervention_latent in always_conditioned_nodes:
            always_conditioned_nodes.remove(intervention_latent)

        ancestors: list[Node] = self.graph.find_ancestors(current_target)
        conditionable_ancestors: list[Node] = []
        for ancestor in ancestors:
            if (
                ancestor.cardinality > 0
                and ancestor.label != current_target.label
                and ancestor not in always_conditioned_nodes
            ):
                conditionable_ancestors.append(ancestor)

        return self._get_possible_d_separator_set(
            conditionable_ancestors,
            always_conditioned_nodes,
            intervention_latent,
            current_target,
            intervention,
        )

    def _get_possible_d_separator_set(
        self,
        conditionable_ancestors: list[Node],
        conditioned_nodes: list[Node],
        intervention_latent: Node,
        current_target: Node,
        intervention: Node,
    ):
        for no_of_possibilities in range(pow(2, len(conditionable_ancestors))):
            current_conditionable_ancestors = self._get_current_conditionable_ancestors(
                conditionable_ancestors, no_of_possibilities
            )
            current_conditioned_nodes: list[Node] = []
            current_conditioned_nodes = (
                conditioned_nodes + current_conditionable_ancestors
            )

            current_conditioned_nodes_labels = [
                node.label for node in current_conditioned_nodes
            ]
            condition1 = nx.is_d_separator(
                G=self.graph.DAG,
                x={current_target.label},
                y={intervention_latent.label},
                z=set(current_conditioned_nodes_labels),
            )

            if intervention in current_conditioned_nodes:
                condition2 = True
            else:
                operated_digraph = copy.deepcopy(self.graph.DAG)
                outgoing_edgesX = list(self.graph.DAG.out_edges(intervention.label))
                operated_digraph.remove_edges_from(outgoing_edgesX)
                condition2 = nx.is_d_separator(
                    G=operated_digraph,
                    x={current_target.label},
                    y={intervention.label},
                    z=set(current_conditioned_nodes_labels),
                )

            if condition1 and condition2:
                valid_d_separator_set: list[Node] = []
                logger.debug("The following set works:")
                for node in current_conditioned_nodes:
                    logger.debug(f"- {node.label}")
                    valid_d_separator_set.append(node)
                break

        if len(valid_d_separator_set) <= 0:
            logger.error("Failure: Could not find a separator set")

        return valid_d_separator_set

    def _get_current_conditionable_ancestors(
        self, conditionable_ancestors: list[Node], no_of_possibilities: int
    ) -> list[Node]:
        current_conditionable_ancestors: list[Node] = []
        for i in range(len(conditionable_ancestors)):
            if (no_of_possibilities >> i) % 2 == 1:
                current_conditionable_ancestors.append(conditionable_ancestors[i])
        return current_conditionable_ancestors

    def generate_symbolic_objective_function_probabilities(self) -> list[tuple]:
        """
        Constructs a symbolic representation of the objective function for the linear program.

        Returns:
            list[tuple]: A list of tuples, where each tuple represents a term in the objective function.
                - For empirical probability variables, the tuple is (node, None). Example: P(A)
                - For conditional probabilities, the tuple is (node, conditioned_nodes),
                where conditioned_nodes is a list of nodes the probability is conditioned on.
                Example: P(A|B)
        """
        objective_function_probabilities: list[tuple] = []
        for node in self.empirical_probabilities_variables:
            if node.is_latent:
                continue
            objective_function_probabilities.append((node, None))
        for node, conditioned_node in self.conditional_probabilities.items():
            objective_function_probabilities.append((node, conditioned_node))
        return objective_function_probabilities
