import copy
import logging

logger = logging.getLogger(__name__)

import networkx as nx
import pandas as pd

from causal_reasoning.do_calculus_algorithm.linear_programming.mechanisms_generator import \
    MechanismGenerator
from causal_reasoning.graph.graph import Graph
from causal_reasoning.graph.node import Node
from causal_reasoning.utils.probabilities_helper import (
    find_conditional_probability, find_probability)
from causal_reasoning.utils.types import MechanismType


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

        self.empiricalProbabilitiesVariables = []
        self.mechanismVariables = []
        self.conditionalProbabilities = []
        self.debugOrder = []

    def get_mechanisms_pruned(self) -> MechanismType:
        """
        Remove c-component variables that do not appear in the objective function
        """
        interventionLatentParent = self.intervention.latentParent
        cComponentEndogenous = interventionLatentParent.children

        endogenousNodes = (set(cComponentEndogenous) & set(self.debugOrder)) | {
            self.intervention
        }

        _, _, mechanisms = MechanismGenerator.mechanisms_generator(
            latentNode=interventionLatentParent,
            endogenousNodes=endogenousNodes,
        )
        return mechanisms

    def build_objective_function(self, mechanisms: MechanismType) -> list[float]:
        """
        Intermediate step: remove useless endogenous variables in the mechanisms creation?
        Must be called after generate restrictions. Returns the objective function with the following encoding

        For each mechanism, find the coefficient in the objective function.
            Open a sum on this.debugOrder variables <=> consider all cases (cartesian product).
            Only the intervention has a fixed value.
        """

        # (3) Generate all the cases: cartesian product!
        debug_variables_label = [node.label for node in self.debugOrder]
        logger.debug(f"Debug variables: {debug_variables_label}")
        if self.intervention in self.debugOrder:
            self.debugOrder.remove(self.intervention)

        summandNodes = list(
            set(self.debugOrder)
            - {
                self.intervention,
                self.intervention.latentParent,
                self.target,
            }
        )

        spaces: list[list[int]] = MechanismGenerator.helper_generate_spaces(
            nodes=summandNodes
        )
        summandNodes.append(self.target)
        spaces.append([self.target.value])
        inputCases: list[list[int]] = MechanismGenerator.generate_cross_products(
            listSpaces=spaces
        )
        """
        TODO: Check the order of "inputCases": it should be the same as the order of the spaces, which is the same as in debugOrder.
        TODO: the case in which the summandNodes is empty (e.g Balke Pearl) has a very ugly fix
        """
        objFunctionCoefficients: list[float] = []
        logger.debug("Debug input cases:")
        logger.debug(f"Size of #inputs: {len(inputCases)}")
        logger.debug("first component:")
        logger.debug(inputCases[0])

        logger.debug("Debug summand nodes")
        for node in summandNodes:
            logger.debug(f"Node={node.label}")

        logger.debug("--- DEBUG OBJ FUNCTION GENERATION ---")
        for mechanism in mechanisms:
            logger.debug("-- START MECHANISM --")
            mechanismCoefficient: int = 0
            for inputCase in inputCases:
                logger.debug("---- START INPUT CASE ----")
                partialCoefficient = 1

                for index, variableValue in enumerate(inputCase):
                    logger.debug(f"{summandNodes[index].label} = {variableValue}")
                    summandNodes[index].value = variableValue

                for variable in summandNodes:
                    logger.debug(f"\nCurrent variable: {variable.label}")
                    if (
                        variable in self.empiricalProbabilitiesVariables
                    ):  # Case 1: coff *= P(V=value)
                        logger.debug("Case 1")
                        variableProbability = find_probability(
                            dataFrame=self.dataFrame,
                            variables=[variable],
                        )
                        partialCoefficient *= variableProbability
                    elif (
                        variable in self.mechanismVariables
                    ):  # Case 2: terminate with coeff 0 if the decision function is 0.
                        # Do nothing otherwise
                        logger.debug("Case 2")
                        current_mechanism_key = []
                        mechanismKey: str = ""
                        for _key, node_item in self.graph.graphNodes.items():
                            if not node_item.isLatent and (
                                variable in node_item.children
                            ):
                                current_mechanism_key.append(
                                    f"{node_item.label}={node_item.value}"
                                )
                        for e in sorted(current_mechanism_key):
                            mechanismKey += f"{e},"
                        logger.debug(f"key: {mechanismKey[:-1]}")
                        expectedValue = mechanism[mechanismKey[:-1]]

                        if expectedValue != variable.value:
                            partialCoefficient = 0
                            logger.debug("End process")
                    else:  # Case 3: coeff *= P(V|some endo parents)
                        logger.debug("Case 3")
                        conditionalProbability = find_conditional_probability(
                            dataFrame=self.dataFrame,
                            targetRealization=[variable],
                            conditionRealization=self.conditionalProbabilities[
                                variable
                            ],
                        )
                        partialCoefficient *= conditionalProbability

                    logger.debug(f"current partial coefficient: {partialCoefficient}")
                    if partialCoefficient == 0:
                        break

                mechanismCoefficient += partialCoefficient
                logger.debug(f"current coef = {mechanismCoefficient}")

            objFunctionCoefficients.append(mechanismCoefficient)

        return objFunctionCoefficients

    def find_linear_good_set(self):
        """
        Runs each step of the algorithm.
        Finds a set of variables/restrictions that linearizes the problem.
        """
        intervention: Node = self.intervention
        current_targets: list[Node] = [self.target]
        interventionLatent: Node = intervention.latentParent

        empiricalProbabilitiesVariables = (
            []
        )  # If V in this array then it implies P(v) in the objective function
        # If V in this array then it implies a decision function: 1(Pa(v) => v=
        # some value)
        mechanismVariables = []
        # If V|A,B,C in this array then it implies P(V|A,B,C) in the objective
        # function
        conditionalProbabilities: dict[Node, list[Node]] = {}
        debugOrder: list[Node] = []
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
            debugOrder.append(current_target)
            logger.debug(f"Current target: {current_target.label}")

            if not self.graph.is_descendant(
                ancestor=self.intervention, descendant=current_target
            ):
                logger.debug("------- Case 1: Not a descendant")
                empiricalProbabilitiesVariables.append(current_target)
            elif current_target.latentParent == interventionLatent:
                logger.debug("------- Case 2: Mechanisms")
                mechanismVariables.append(current_target)
                for parent in current_target.parents:
                    if (parent not in current_targets) and parent != intervention:
                        current_targets.append(parent)
            else:
                logger.debug("------- Case 3: Find d-separator set")
                valid_d_separator_set = self._find_d_separator_set(
                    current_target, current_targets, interventionLatent, intervention
                )

                current_targets = list(
                    (set(current_targets) | set(valid_d_separator_set))
                    - {current_target}
                )

                conditionalProbabilities[current_target] = valid_d_separator_set

                # Question: is any already solved variable selected for the second time? Does the program need to address this issue
                # by forcing the set to not contain any of such variables?

        self.empiricalProbabilitiesVariables = empiricalProbabilitiesVariables
        self.mechanismVariables = mechanismVariables
        self.conditionalProbabilities = conditionalProbabilities
        self.debugOrder = debugOrder
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
            # Question: does it need to not be the intervention?
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
        # testa todas as possibilidades de condicionar conjuntos de variáveis nesse vetor
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
                operatedDigraph = copy.deepcopy(self.graph.DAG)
                outgoing_edgesX = list(self.graph.DAG.out_edges(intervention.label))
                operatedDigraph.remove_edges_from(outgoing_edgesX)
                condition2 = nx.is_d_separator(
                    G=operatedDigraph,
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

        # Returns one of the valid subsets - Last instance of
        # "valid_d_separator_set", for now.
        return valid_d_separator_set

    def _get_current_conditionable_ancestors(
        self, conditionable_ancestors: list[Node], no_of_possibilities: int
    ) -> list[Node]:
        current_conditionable_ancestors: list[Node] = []
        for i in range(len(conditionable_ancestors)):
            if (no_of_possibilities >> i) % 2 == 1:
                current_conditionable_ancestors.append(conditionable_ancestors[i])
        return current_conditionable_ancestors
