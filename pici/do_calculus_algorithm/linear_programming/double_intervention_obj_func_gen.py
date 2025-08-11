import copy
import logging

logger = logging.getLogger(__name__)

import networkx as nx
import pandas as pd

from pici.do_calculus_algorithm.linear_programming.mechanisms_generator import (
    MechanismGenerator,
)
from pici.graph.graph import Graph
from pici.graph.node import Node
from pici.utils.probabilities_helper import (
    find_conditional_probability,
    find_probability,
)
from pici.utils.types import MechanismType


class DoubleInterventionObjFunctionGenerator:
    """
    Given an intervention and a graph, this class finds a set of restrictions that can be used to build
    a linear objective function.
    """

    def __init__(
        self,
        graph: Graph,
        dataFrame: pd.DataFrame,
        interventions: tuple[Node, Node],
        target: Node,
    ):
        """
        graph: an instance of the personalized class graph
        intervention: X in P(Y|do(X))
        target: Y in P(Y|do(X))
        """

        self.graph = graph
        self.intervention_1 = interventions[0]
        self.intervention_2 = interventions[1]
        self.target = target
        self.dataFrame = dataFrame

        self.empiricalProbabilitiesVariables = []
        self.mechanismVariables_1 = []
        self.mechanismVariables_2 = []
        self.conditionalProbabilities = []
        self.debugOrder = []

    def get_mechanisms_pruned(self) -> tuple[MechanismType, MechanismType]:
        """
        Remove c-component variables that do not appear in the objective function
        """
        interventionLatentParent_1 = self.intervention_1.latentParent
        cComponentEndogenous_1 = interventionLatentParent_1.children

        endogenousNodes_1 = (set(cComponentEndogenous_1) & set(self.debugOrder)) | {
            self.intervention_1
        }

        _, _, mechanisms_1 = MechanismGenerator.mechanisms_generator(
            latentNode=interventionLatentParent_1,
            endogenousNodes=endogenousNodes_1,
        )

        interventionLatentParent_2 = self.intervention_2.latentParent
        cComponentEndogenous_2 = interventionLatentParent_2.children

        endogenousNodes_2 = (set(cComponentEndogenous_2) & set(self.debugOrder)) | {
            self.intervention_2
        }

        _, _, mechanisms_2 = MechanismGenerator.mechanisms_generator(
            latentNode=interventionLatentParent_2,
            endogenousNodes=endogenousNodes_2,
        )
        return mechanisms_1, mechanisms_2

    def build_objective_function(
        self, mechanisms_1: MechanismType, mechanisms_2: MechanismType
    ) -> list[float]:
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
        if self.intervention_1 in self.debugOrder:
            self.debugOrder.remove(self.intervention_1)

        if self.intervention_2 in self.debugOrder:
            self.debugOrder.remove(self.intervention_2)

        summandNodes = list(
            set(self.debugOrder)
            - {
                self.intervention_1,
                self.intervention_1.latentParent,
                self.intervention_2,
                self.intervention_2.latentParent,
                self.target,
            }
        )

        spaces: list[list[int]] = MechanismGenerator.helper_generate_spaces(
            nodes=summandNodes
        )
        summandNodes.append(self.target)
        spaces.append([self.target.intervened_value])
        inputCases: list[list[int]] = MechanismGenerator.generate_cross_products(
            listSpaces=spaces
        )
        """
        TODO: Check the order of "inputCases": it should be the same as the order of the spaces, which is the same as in debugOrder.
        TODO: the case in which the summandNodes is empty (e.g Balke Pearl) has a very ugly fix
        """

        # Pensar se é melhor uni ou bi dimensional
        obj_function_coefficients: list[list[float]] = []
        for i in range(len(mechanisms_1)):
            obj_function_coefficients.append([])
            for j in range(len(mechanisms_2)):
                obj_function_coefficients[i].append(0)

        # INICIALIZAR com len(u) e len(w)
        # [
        # [9 8 7 6 5]
        # [0 1 2 3 4]
        # ]

        # i, j
        # c_0_0 u[0] w[0] 9

        logger.debug("Debug input cases:")
        logger.debug(f"Size of #inputs: {len(inputCases)}")
        logger.debug("first component:")
        logger.debug(inputCases[0])

        logger.debug("Debug summand nodes")
        for node in summandNodes:
            logger.debug(f"Node={node.label}")

        logger.debug("--- DEBUG OBJ FUNCTION GENERATION ---")
        for i, mechanism_1 in enumerate(mechanisms_1):
            for j, mechanism_2 in enumerate(mechanisms_2):
                # (sum w e W sum u e U c_u_w P(u) P(w))
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
                            variable in self.mechanismVariables_1
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
                            expectedValue = mechanism_1[mechanismKey[:-1]]

                            if expectedValue != variable.value:
                                partialCoefficient = 0
                                logger.debug("End process")
                        elif (
                            variable in self.mechanismVariables_2
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
                            expectedValue = mechanism_2[mechanismKey[:-1]]

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

                        logger.debug(
                            f"current partial coefficient: {partialCoefficient}"
                        )
                        if partialCoefficient == 0:
                            break

                    mechanismCoefficient += partialCoefficient
                    logger.debug(f"current coef = {mechanismCoefficient}")
                obj_function_coefficients[i][j] = mechanismCoefficient
        return obj_function_coefficients

    def find_linear_good_set(self):
        """
        Runs each step of the algorithm.
        Finds a set of variables/restrictions that linearizes the problem.
        """
        intervention_1: Node = self.intervention_1
        intervention_2: Node = self.intervention_2
        current_targets: list[Node] = [self.target]
        interventionLatent_1: Node = intervention_1.latentParent
        interventionLatent_2: Node = intervention_2.latentParent

        empiricalProbabilitiesVariables = (
            []
        )  # If V in this array then it implies P(v) in the objective function
        # If V in this array then it implies a decision function: 1(Pa(v) => v=
        # some value)
        mechanismVariables_1 = []
        mechanismVariables_2 = []
        # If V|A,B,C in this array then it implies P(V|A,B,C) in the objective
        # function
        conditionalProbabilities: dict[Node, list[Node]] = {}
        debugOrder: list[Node] = []
        while len(current_targets) > 0:
            logger.debug("---- Current targets array:")
            for tg in current_targets:
                logger.debug(f"- {tg.label}")

            # ARROYO-> TODO: check if the topological order is reversed.
            current_target = (
                self.graph.get_closest_node_from_leaf_in_the_topological_order(
                    current_targets
                )
            )
            logger.debug(f"__>>{current_target.label}<<__")
            if current_target in current_targets:
                current_targets.remove(current_target)
            debugOrder.append(current_target)
            logger.debug(f"Current target: {current_target.label}")

            if not self.graph.is_descendant(
                ancestor=self.intervention_1, descendant=current_target
            ) and not self.graph.is_descendant(
                ancestor=self.intervention_2, descendant=current_target
            ):
                logger.debug("------- Case 1: Not a descendant")
                empiricalProbabilitiesVariables.append(current_target)
            elif current_target.latentParent == interventionLatent_1:
                logger.debug("------- Case 2: Mechanisms Variables 1")
                mechanismVariables_1.append(current_target)
                for parent in current_target.parents:
                    if (parent not in current_targets) and parent != intervention_1:
                        current_targets.append(parent)
            elif current_target.latentParent == interventionLatent_2:
                logger.debug("------- Case 2: Mechanisms Variables 2")
                mechanismVariables_2.append(current_target)
                for parent in current_target.parents:
                    if (parent not in current_targets) and parent != intervention_2:
                        current_targets.append(parent)
            else:
                # Acrescentar checagens
                logger.debug("------- Case 3: Find d-separator set")
                validConditionedNodes = self._find_d_separator_set(
                    current_target,
                    current_targets,
                    interventionLatent_1,
                    interventionLatent_2,
                    intervention_1,
                    intervention_2,
                )

                current_targets = list(
                    (set(current_targets) | set(validConditionedNodes))
                    - {intervention_1, intervention_2, current_target}
                )

                conditionalProbabilities[current_target] = validConditionedNodes

                # Question: is any already solved variable selected for the second time? Does the program need to address this issue
                # by forcing the set to not contain any of such variables?

        self.empiricalProbabilitiesVariables = empiricalProbabilitiesVariables
        self.mechanismVariables_1 = mechanismVariables_1
        self.mechanismVariables_2 = mechanismVariables_2
        self.conditionalProbabilities = conditionalProbabilities
        self.debugOrder = debugOrder
        logger.debug("Test completed")

    def _find_d_separator_set(
        self,
        current_target: Node,
        current_targets: list[Node],
        intervention_latent_1: Node,
        intervention_latent_2: Node,
        intervention_1: Node,
        intervention_2: Node,
    ):

        always_conditioned_nodes: list[Node] = current_targets.copy()
        if current_target in always_conditioned_nodes:
            always_conditioned_nodes.remove(current_target)

        if intervention_latent_1 in always_conditioned_nodes:
            always_conditioned_nodes.remove(intervention_latent_1)

        if intervention_latent_2 in always_conditioned_nodes:
            always_conditioned_nodes.remove(intervention_latent_2)

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
            intervention_latent_1,
            intervention_latent_2,
            current_target,
            intervention_1,
            intervention_2,
        )

    def _get_possible_d_separator_set(
        self,
        conditionable_ancestors: list[Node],
        conditioned_nodes: list[Node],
        intervention_latent_1: Node,
        intervention_latent_2: Node,
        current_target: Node,
        intervention_1: Node,
        intervention_2: Node,
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

            # self.graph.build_moral(
            #     consideredNodes=ancestors, conditionedNodes=current_conditioned_nodes
            # )
            # condition1_1 = self.graph.independency_moral(
            #     node_2=intervention_latent_1, node_1=current_target
            # )
            # condition1_2 = self.graph.independency_moral(
            #     node_2=intervention_latent_2, node_1=current_target
            # )
            intervention_1_first_condition = nx.is_d_separator(
                G=self.graph.DAG,
                x={current_target.label},
                y={intervention_latent_1.label},
                z=set(current_conditioned_nodes_labels),
            )

            if intervention_1 in current_conditioned_nodes:
                intervention_1_second_condition = True
            else:
                operatedDigraph = copy.deepcopy(self.graph.DAG)
                outgoing_edgesX = list(self.graph.DAG.out_edges(intervention_1.label))
                operatedDigraph.remove_edges_from(outgoing_edgesX)
                intervention_1_second_condition = nx.is_d_separator(
                    G=operatedDigraph,
                    x={current_target.label},
                    y={intervention_1.label},
                    z=set(current_conditioned_nodes_labels),
                )

            # self.graph.build_moral(
            #     consideredNodes=ancestors,
            #     conditionedNodes=current_conditioned_nodes,
            #     intervention_outgoing_edges_are_considered=False,
            #     intervention=intervention_1,
            # )
            # condition2_1 = self.graph.independency_moral(
            #     node_2=intervention_1, node_1=current_target
            # )

            # self.graph.build_moral(
            #     consideredNodes=ancestors,
            #     conditionedNodes=current_conditioned_nodes,
            #     intervention_outgoing_edges_are_considered=False,
            #     intervention=intervention_2,
            # )
            # condition2_2 = self.graph.independency_moral(
            #     node_2=intervention_2, node_1=current_target
            # )
            intervention_2_first_condition = nx.is_d_separator(
                G=self.graph.DAG,
                x={current_target.label},
                y={intervention_latent_2.label},
                z=set(current_conditioned_nodes_labels),
            )

            if intervention_2 in current_conditioned_nodes:
                intervention_2_second_condition = True
            else:
                operatedDigraph = copy.deepcopy(self.graph.DAG)
                outgoing_edgesX = list(self.graph.DAG.out_edges(intervention_2.label))
                operatedDigraph.remove_edges_from(outgoing_edgesX)
                intervention_2_second_condition = nx.is_d_separator(
                    G=operatedDigraph,
                    x={current_target.label},
                    y={intervention_2.label},
                    z=set(current_conditioned_nodes_labels),
                )

            if (
                intervention_1_first_condition
                and intervention_1_second_condition
                and intervention_2_first_condition
                and intervention_2_second_condition
            ):
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
