import pandas as pd
from scipy.optimize import linprog
import logging

logger = logging.getLogger(__name__)

from causal_reasoning.graph.graph import Graph
from causal_reasoning.graph.node import Node
from causal_reasoning.linear_algorithm.linear_constraints import generate_constraints
from causal_reasoning.linear_algorithm.obj_function_generator import (
    ObjFunctionGenerator,
)


def builder_linear_problem(
    graph: Graph,
    df: pd.DataFrame,
    intervention: Node,
    target: Node,
):
    objFG = ObjFunctionGenerator(
        graph=graph,
        dataFrame=df,
        intervention=intervention,
        target=target,
    )
    objFG.find_linear_good_set() # Qual parte do grafo a gente vai considerar / Quais Nodes serão considerados? 
    mechanisms = objFG.get_mechanisms_pruned() 
    # Quais são as variáveis exógenas que serão consideradas?
    # Com isso pegamos o número de variáveis que as exógenas podem assumir.

    interventionLatentParent = objFG.intervention.latentParent
    cComponentEndogenous = interventionLatentParent.children
    consideredEndogenousNodes = list(
        (set(cComponentEndogenous) & set(objFG.debugOrder)) | {objFG.intervention}
    )

    probs, decisionMatrix = generate_constraints(
        data=df,
        dag=objFG.graph,
        unob=interventionLatentParent,
        consideredCcomp=consideredEndogenousNodes,
        mechanisms=mechanisms,
    )

    objFunctionCoefficients: list[float] = objFG.build_objective_function(mechanisms)
    logger.debug("-- DEBUG OBJ FUNCTION --")
    for i, coeff in enumerate(objFunctionCoefficients):
        logger.debug(f"c_{i} = {coeff}")

    logger.debug("-- DECISION MATRIX --")
    for i in range(len(decisionMatrix)):
        for j in range(len(decisionMatrix[i])):
            logger.debug(f"{decisionMatrix[i][j]} ", end="")
        logger.debug(f" = {probs[i]}")
    intervals = [(0, 1) for _ in range(len(decisionMatrix[0]))]

    # lowerBound, upperBound = causal_reasoning.column_generation.pyomo_use(objFunctionCoefficients, decisionMatrix, probs, intervals)

    lowerBoundSol = linprog(
        c=objFunctionCoefficients,
        A_ub=None,
        b_ub=None,
        A_eq=decisionMatrix,
        b_eq=probs,
        method="highs",
        bounds=intervals,
    )
    lowerBound = lowerBoundSol.fun

    upperBoundSol = linprog(
        c=[-x for x in objFunctionCoefficients],
        A_ub=None,
        b_ub=None,
        A_eq=decisionMatrix,
        b_eq=probs,
        method="highs",
        bounds=intervals,
    )
    upperBound = -upperBoundSol.fun

    logger.debug(
        f"Causal query: P({target.label}={target.value}|do({intervention.label}={intervention.value}))"
    )
    logger.debug(f"Bounds: {lowerBound} <= P <= {upperBound}")
