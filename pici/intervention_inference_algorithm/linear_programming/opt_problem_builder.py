import logging

import pandas as pd

from pici.intervention_inference_algorithm.column_generation.generic.subproblem import get_node_list_realizations
from pici.utils.probabilities_helper import find_conditional_probability

logger = logging.getLogger(__name__)

from pici.graph.graph import Graph
from pici.graph.node import Node
from pici.intervention_inference_algorithm.linear_programming.linear_constraints import (
    generate_constraints,
)
from pici.intervention_inference_algorithm.linear_programming.obj_function_generator import (
    ObjFunctionGenerator,
)
from pici.intervention_inference_algorithm.linear_programming.optimizers import (
    Optimizer,
    choose_optimizer,
    compute_bounds,
)
from pici.utils._enum import OptimizersLabels


def build_linear_problem(
    graph: Graph,
    df: pd.DataFrame,
    intervention: Node,
    target: Node,
    optimizer_label: str = OptimizersLabels.GUROBI.value,
) -> tuple[str, str]:
    objFG = ObjFunctionGenerator(
        graph=graph,
        dataFrame=df,
        intervention=intervention,
        target=target,
    )
    ##
    # TODO: REMOVER APÓS A FINALIZAÇÃO
    # symbolic = objFG.generate_symbolic_objective_function_probabilities()
    # list_node = set()
    # for s in symbolic:
    #     list_node.add(s[0])
    #     for n in s[1]:
    #         list_node.add(n)
    # list_node = list(list_node)
    # cartesian_product = get_node_list_realizations(list_node)
    # header = cartesian_product[0]
    # cartesian_product =cartesian_product[1:]
    
    # for product in cartesian_product:
    #     for s in symbolic:
    #         s[0].value = product[header.index(s[0].label)]
    #         str_p = ""
    #         for node in s[1]:
    #             node.value = product[header.index(node.label)]
    #             str_p = f"{node.label}={node.value}, "
    #         c = find_conditional_probability(dataFrame=df,target_realization=[s[0]], condition_realization=s[1])
    #         print(f"P({s[0].label}={s[0].value} | {str_p}) = {c}")
    # print("-------------------------")
    
    mechanisms = objFG.get_mechanisms_pruned()

    interventionLatentParent = objFG.intervention.latent_parent
    cComponentEndogenous = interventionLatentParent.children
    consideredEndogenousNodes = list(
        (set(cComponentEndogenous) & set(objFG.considered_graph_nodes))
        | {objFG.intervention}
    )

    probs, decision_matrix = generate_constraints(
        data=df,
        dag=objFG.graph,
        unob=interventionLatentParent,
        considered_c_comp=consideredEndogenousNodes,
        mechanisms=mechanisms,
    )

    intervention.value = intervention.intervened_value
    obj_function_coefficients: list[float] = objFG.build_objective_function(mechanisms)

    logger.debug("-- DEBUG OBJ FUNCTION --")
    for i, coeff in enumerate(obj_function_coefficients):
        logger.debug(f"c_{i} = {coeff}")

    logger.debug("-- DECISION MATRIX --")
    for i in range(len(decision_matrix)):
        for j in range(len(decision_matrix[i])):
            logger.debug(f"{decision_matrix[i][j]} ")
        logger.debug(f" = {probs[i]}")

    optimizer: Optimizer = choose_optimizer(
        optimizer_label,
        probs=probs,
        decision_matrix=decision_matrix,
        obj_function_coefficients=obj_function_coefficients,
    )

    lowerBound, upperBound = compute_bounds(optimizer)

    logger.info(
        f"Causal query: P({target.label}={target.intervened_value}|do({intervention.label}={intervention.intervened_value}))"
    )
    logger.info(f"Bounds: {lowerBound} <= P <= {upperBound}")
    return lowerBound, upperBound
