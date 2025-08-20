import logging

import pandas as pd

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
    mechanisms = objFG.get_mechanisms_pruned()

    interventionLatentParent = objFG.intervention.latentParent
    cComponentEndogenous = interventionLatentParent.children
    consideredEndogenousNodes = list(
        (set(cComponentEndogenous) & set(objFG.consideredGraphNodes))
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
