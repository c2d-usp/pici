import pandas as pd

from pici.do_calculus_algorithm.linear_programming.mechanisms_generator import (
    MechanismGenerator,
)
from pici.graph.graph import Graph
from pici.graph.node import Node
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
    cCompOrder: list[Node] = []
    probs: list[float] = [1.0]
    condVars: list[Node] = []
    usedVars: list[Node] = []
    productTerms: list[dict[Node, list[Node]]] = []

    decision_matrix: list[list[int]] = [[1 for _ in range(len(mechanisms))]]

    for node in topoOrder:
        if (unob in node.parents) and (node in consideredCcomp):
            cCompOrder.append(node)
    cCompOrder.reverse()
    usedVars = cCompOrder.copy()
    # TODO: O QUE QUE É WC? --> PAIS + TAIL
    Wc: list[Node] = cCompOrder.copy()
    for cCompNode in cCompOrder:
        for par in cCompNode.parents:
            if par not in Wc and (par != unob):
                Wc.append(par)

    # TODO: ENQUANTO NÃO ESTIVER VAZIA
    while bool(cCompOrder):
        node = cCompOrder.pop(0)
        for cond in Wc:
            if topoOrder.index(cond) < topoOrder.index(node):
                if cond not in condVars:
                    condVars.append(cond)
                if cond not in usedVars:
                    usedVars.append(cond)
        productTerms.append({node: condVars.copy()})
        condVars.clear()
    spaces: list[list[int]] = [range(var.cardinality) for var in usedVars]
    cartesianProduct: list[list[int]] = MechanismGenerator.generate_cross_products(
        listSpaces=spaces
    )

    for rlt in cartesianProduct:
        prob = 1.0
        # TODO: TRANSFORMAR EM FUNÇÃO ISSO AQUI:
        for term in productTerms:
            targetRealizationNodes: list[Node] = []
            conditionRealizationNodes: list[Node] = []
            for key_node in term:
                key_node.value = rlt[usedVars.index(key_node)]
                targetRealizationNodes.append(key_node)
                for cVar in term[key_node]:
                    cVar.value = rlt[usedVars.index(cVar)]
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
            for var in usedVars:
                if var in consideredCcomp:
                    endoParents: list[Node] = var.parents.copy()
                    endoParents.remove(unob)
                    # TODO: key é algo do tipo 'A=1,B=3'
                    key = create_dict_index(
                        parents=endoParents, rlt=rlt, indexerList=usedVars
                    )
                    endoParents.clear()
                    if mechanisms[u][key] == rlt[usedVars.index(var)]:
                        coef *= 1
                    else:
                        coef *= 0
                        break
            aux.append(float(coef))
        decision_matrix.append(aux)
    return probs, decision_matrix
