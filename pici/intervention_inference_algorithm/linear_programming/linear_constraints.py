import pandas as pd

from pici.graph.graph import Graph
from pici.graph.node import Node
from pici.intervention_inference_algorithm.linear_programming.mechanisms_generator import (
    MechanismGenerator,
)
from pici.utils.probabilities_helper import find_conditional_probability
from pici.utils.types import MechanismType


def create_dict_index(parents: list[Node], rlt: list[int], indexerList: list[Node]):
    current_index = []
    for parNode in parents:
        current_index.append(
            str(parNode.label) + "=" + str(rlt[indexerList.index(parNode)])
        )
    index: str = ""
    for e in sorted(current_index):
        index += f"{e},"
    return index[:-1]


def generate_constraints(
    data: pd.DataFrame,
    dag: Graph,
    unob: Node,
    consideredCcomp: list[Node],
    mechanisms: MechanismType,
) -> tuple[list[float], list[list[int]]]:
    topoOrder: list[Node] = dag.topologicalOrder
    cCompOrder = get_c_component_in_reverse_topological_order(
        topoOrder, unob, consideredCcomp
    )
    c_component_and_tail: list[Node] = find_c_component_and_tail_set(unob, cCompOrder)

    symbolical_constraints_probabilities, Wc = (
        get_symbolical_constraints_probabilities_and_wc(
            cCompOrder, c_component_and_tail, topoOrder
        )
    )

    probs, decision_matrix = calculate_numerical_constraints(
        data=data,
        mechanisms=mechanisms,
        Wc=Wc,
        symbolical_constraints_probabilities=symbolical_constraints_probabilities,
        consideredCcomp=consideredCcomp,
        unob=unob,
    )
    return probs, decision_matrix


def get_c_component_in_reverse_topological_order(
    topoOrder: list[Node], unob: Node, consideredCcomp: list[Node]
) -> list[Node]:
    cCompOrder: list[Node] = []
    for node in topoOrder:
        if (unob in node.parents) and (node in consideredCcomp):
            cCompOrder.append(node)
    cCompOrder.reverse()
    return cCompOrder


def find_c_component_and_tail_set(unob: Node, cCompOrder: list[Node]) -> list[Node]:
    """
    Find Conjunto de nós Ccomp + Pais, ordenado em ordem topológica
    """
    c_component_and_tail: list[Node] = cCompOrder.copy()
    for cCompNode in cCompOrder:
        for par in cCompNode.parents:
            if par not in c_component_and_tail and (par != unob):
                c_component_and_tail.append(par)
    return c_component_and_tail


def get_symbolical_constraints_probabilities_and_wc(
    cCompOrder: list[Node], c_component_and_tail: list[Node], topoOrder: list[Node]
) -> tuple[list[dict[Node, list[Node]]], list[Node]]:
    """
    Wc é um subconjunto do cComp + Tail o Wc contém todas as variáveis presentes nas restrições do problema
    """
    condVars: list[Node] = []
    symbolical_constraints_probabilities: list[dict[Node, list[Node]]] = []
    Wc: list[Node] = []
    Wc = cCompOrder.copy()
    while bool(cCompOrder):
        node = cCompOrder.pop(0)
        for cond in c_component_and_tail:
            if topoOrder.index(cond) < topoOrder.index(node):
                if cond not in condVars:
                    condVars.append(cond)
                if cond not in Wc:
                    Wc.append(cond)
        symbolical_constraints_probabilities.append({node: condVars.copy()})
        condVars.clear()
    return symbolical_constraints_probabilities, Wc


def calculate_numerical_constraints(
    data: pd.DataFrame,
    mechanisms: MechanismType,
    Wc: list[Node],
    symbolical_constraints_probabilities: list[dict[Node, list[Node]]],
    consideredCcomp: list[Node],
    unob: Node,
) -> tuple[list[float], list[list[int]]]:
    probs: list[float] = [1.0]
    decision_matrix: list[list[int]] = [[1 for _ in range(len(mechanisms))]]
    spaces: list[list[int]] = [range(var.cardinality) for var in Wc]
    cartesianProduct: list[list[int]] = MechanismGenerator.generate_cross_products(
        listSpaces=spaces
    )
    for rlt in cartesianProduct:
        prob = 1.0
        for term in symbolical_constraints_probabilities:
            targetRealizationNodes: list[Node] = []
            conditionRealizationNodes: list[Node] = []
            for key_node in term:
                key_node.value = rlt[Wc.index(key_node)]
                targetRealizationNodes.append(key_node)
                for cVar in term[key_node]:
                    cVar.value = rlt[Wc.index(cVar)]
                    conditionRealizationNodes.append(cVar)
            prob *= find_conditional_probability(
                dataFrame=data,
                targetRealization=targetRealizationNodes,
                conditionRealization=conditionRealizationNodes,
            )
            targetRealizationNodes.clear()
            conditionRealizationNodes.clear()

        probs.append(prob)
        aux: list[int] = []
        for u in range(len(mechanisms)):
            coef: bool = True
            for var in Wc:
                if var in consideredCcomp:
                    endoParents: list[Node] = var.parents.copy()
                    endoParents.remove(unob)
                    key = create_dict_index(
                        parents=endoParents, rlt=rlt, indexerList=Wc
                    )
                    endoParents.clear()
                    if mechanisms[u][key] == rlt[Wc.index(var)]:
                        coef *= 1
                    else:
                        coef *= 0
                        break
            aux.append(float(coef))
        decision_matrix.append(aux)
    return probs, decision_matrix
