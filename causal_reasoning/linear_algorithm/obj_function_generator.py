from causal_reasoning.graph.graph import Graph
from causal_reasoning.graph.node import T, Node
from causal_reasoning.linear_algorithm.mechanisms_generator import MechanismGenerator
from causal_reasoning.linear_algorithm.probabilities_helper import find_probability, find_conditional_probability


class ObjFunctionGenerator:
    """
    Given an intervention and a graph, this class finds a set of restrictions that can be used to build
    a linear objective function.
    """

    def __init__(
        self,
        graph: Graph,
        intervention: Node,
        target: Node,
        intervention_value: int,
        target_value: int,
        dataFrame,
        empiricalProbabilitiesVariables: list[Node],
        mechanismVariables: list[Node],
        conditionalProbabilities: dict[Node, list[Node]],
        debugOrder: list[Node],
    ):
        """
        graph: an instance of the personalized class graph
        intervention: X in P(Y|do(X))
        intervention_value: the value assumed by the X variable
        target: Y in P(Y|do(X))
        """

        self.graph = graph
        self.intervention = intervention
        self.intervention_value = intervention_value
        self.target = target
        self.target_value = target_value
        self.dataFrame = dataFrame

        self.empiricalProbabilitiesVariables = empiricalProbabilitiesVariables
        self.mechanismVariables = mechanismVariables
        self.conditionalProbabilities = conditionalProbabilities
        self.debugOrder = debugOrder

    def find_linear_good_set(self):
        """
        Runs each step of the algorithm. Finds a set of variables/restrictions that linearizes the problem.
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
            print("---- Current targets array:")
            for tg in current_targets:
                print(f"- {tg.label}")

            # ARROYO-> TODO: check if the topological order is reversed.
            current_target = self.graph.get_closest_node_from_leaf_in_the_topological_order(current_targets)
            print(f'__>>{current_target}<<__')
            if current_target in current_targets: current_targets.remove(current_target)
            debugOrder.append(current_target)
            print(f"Current target: {current_target}")

            if not self.graph.is_descendant(
                ancestor=self.intervention, descendant=current_target
            ):
                print(f"------- Case 1: Not a descendant")
                empiricalProbabilitiesVariables.append(current_target)
            elif (
                current_target.latentParent == interventionLatent
            ):
                print(f"------- Case 2: Mechanisms")
                mechanismVariables.append(current_target)
                for parent in current_target.parents:
                    if (parent not in current_targets) and parent != intervention:
                        current_targets.append(parent)
            else:
                print(f"------- Case 3: Find d-separator set")
                validConditionedNodes = self.find_d_separator_set(
                    current_target, current_targets,
                    interventionLatent, intervention
                )

                current_targets = list(
                    (set(current_targets) | set(validConditionedNodes))
                    - {intervention, current_target}
                )

                conditionalProbabilities[current_target] = validConditionedNodes

                # Question: is any already solved variable selected for the second time? Does the program need to address this issue
                # by forcing the set to not contain any of such variables?

        self.empiricalProbabilitiesVariables = empiricalProbabilitiesVariables
        self.mechanismVariables = mechanismVariables
        self.conditionalProbabilities = conditionalProbabilities
        self.debugOrder = debugOrder
        print("Test completed")


    def find_d_separator_set(
        self,
        current_target: Node,
        current_targets: list[Node],
        intervention_latent: Node,
        intervention: Node
    ):
        ancestors: list[Node] = self.graph.find_ancestors(current_target)
        conditionable_ancestors: list[Node] = []

        for ancestor in ancestors:
            # Question: does it need to not be the intervention?
            if (
                ancestor.cardinality > 0
                and ancestor.label != current_target.label
            ):
                conditionable_ancestors.append(ancestor)

        always_conditioned_nodes: list[Node] = current_targets.copy()
        if current_target in always_conditioned_nodes:
            always_conditioned_nodes.remove(current_target)

        if intervention_latent in always_conditioned_nodes:
            always_conditioned_nodes.remove(intervention_latent)
        
        return self.test_all_conditioned_sets(
            conditionable_ancestors,
            always_conditioned_nodes,
            ancestors,
            intervention_latent,
            current_target,
            intervention
        )


    # TODO: I suggest rename it
    def test_all_conditioned_sets(
        self,
        conditionable_ancestors: list[Node],
        conditioned_nodes: list[Node],
        ancestors: list[Node],
        intervention_latent: Node,
        current_target: Node,
        intervention: Node
    ):
        # testa todas as possibilidades de condicionar conjuntos de variáveis nesse vetor
        for no_of_possibilities in range(pow(2, len(conditionable_ancestors))):
            # CRIAR UMA FUNÇÃO PARA ISSO
            for i in range(len(conditionable_ancestors)):
                if (no_of_possibilities >> i) % 2 == 1:
                    conditioned_nodes.append(conditionable_ancestors[i])

            self.graph.build_moral(
                consideredNodes=ancestors, conditionedNodes=conditioned_nodes
            )
            condition1 = self.graph.independency_moral(
                node2=intervention_latent, node1=current_target
            )

            self.graph.build_moral(
                consideredNodes=ancestors,
                conditionedNodes=conditioned_nodes,
                intervention_outgoing_edges_are_considered=False,
                intervention=intervention,
            )
            condition2 = self.graph.independency_moral(
                node2=intervention, node1=current_target
            )
            if condition1 and condition2:
                valid_conditioned_nodes: list[Node] = []
                print(f"The following set works:")
                for node in conditioned_nodes:
                    print(f"- {node.label}")
                    valid_conditioned_nodes.append(node)
        # Returns one of the valid subsets - Last instance of
        # "valid_conditioned_nodes", for now.
        return valid_conditioned_nodes


    def get_mechanisms_pruned(self) -> list[dict[T, int]]:
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

    def build_objective_function(self, mechanisms: list[dict[T, int]]) -> list[float]:
        """
        Intermediate step: remove useless endogenous variables in the mechanisms creation?
        Must be called after generate restrictions. Returns the objective function with the following encoding

        For each mechanism, find the coefficient in the objective function.
            Open a sum on this.debugOrder variables <=> consider all cases (cartesian product).
            Only the intervention has a fixed value.
        """

        # (3) Generate all the cases: cartesian product!
        debug_variables_label = [node.label for node in self.debugOrder]
        print(f"Debug variables: {debug_variables_label}")
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
        spaces.append([self.target_value])
        inputCases: list[list[int]] = MechanismGenerator.generate_cross_products(
            listSpaces=spaces
        )
        """
        TODO: Check the order of "inputCases": it should be the same as the order of the spaces, which is the same as in debugOrder.
        TODO: the case in which the summandNodes is empty (e.g Balke Pearl) has a very ugly fix
        """
        objFunctionCoefficients: list[float] = []
        print("Debug input cases:")
        print(f"Size of #inputs: {len(inputCases)}")
        print(f"first component:")
        print(inputCases[0])

        print("Debug summand nodes")
        for node in summandNodes:
            print(f"Node={node.label}")

        print("--- DEBUG OBJ FUNCTION GENERATION ---")
        for mechanism in mechanisms:
            print("-- START MECHANISM --")
            mechanismCoefficient: int = 0
            for inputCase in inputCases:
                print("---- START INPUT CASE ----")
                # TODO: POSSO FAZER ESSA ATRIBUIÇÃO ANTES NO CÓDIGO
                self.intervention.value = self.intervention_value
                self.target.value = self.target_value,
                partialCoefficient = 1

                for index, variableValue in enumerate(inputCase):
                    print(
                        f"{summandNodes[index].label} = {variableValue}",
                        end="",
                    )
                    summandNodes[index].value = variableValue

                for variable in summandNodes:
                    print(
                        f"\nCurrent variable: {variable.label}"
                    )
                    if (
                        variable in self.empiricalProbabilitiesVariables
                    ):  # Case 1: coff *= P(V=value)
                        print("Case 1")
                        variableProbability = find_probability(
                            dataFrame=self.dataFrame,
                            variables=[variable],
                        )
                        partialCoefficient *= variableProbability
                    elif (
                        variable in self.mechanismVariables
                    ):  # Case 2: terminate with coeff 0 if the decision function is 0. Do nothing otherwise
                        print("Case 2")
                        mechanismKey: str = ""
                        for _key, node_item in self.graph.graphNodes.items():
                            if not node_item.isLatent and (variable in node_item.children):
                                mechanismKey += (
                                    f"{node_item.label}={node_item.value},"
                                )
                        print(f"key: {mechanismKey[:-1]}")
                        expectedValue = mechanism[mechanismKey[:-1]]

                        if expectedValue != variable.value:
                            partialCoefficient = 0
                            print("End process")
                    else:  # Case 3: coeff *= P(V|some endo parents)
                        print("Case 3")
                        conditionalProbability = (
                            find_conditional_probability(
                                dataFrame=self.dataFrame,
                                targetRealization=[variable.value],
                                conditionRealization=self.conditionalProbabilities[variable],
                            )
                        )
                        partialCoefficient *= conditionalProbability

                    print(f"current partial coefficient: {partialCoefficient}")
                    if partialCoefficient == 0:
                        break

                mechanismCoefficient += partialCoefficient
                print(f"current coef = {mechanismCoefficient}")

            objFunctionCoefficients.append(mechanismCoefficient)

        return objFunctionCoefficients
