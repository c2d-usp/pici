import pandas as pd
from scipy.optimize import linprog
import logging
import gurobipy as gp

from causal_reasoning.linear_algorithm.dual_obj_func_gen import DoubleObjFunctionGenerator

logger = logging.getLogger(__name__)

from causal_reasoning.graph.graph import Graph
from causal_reasoning.graph.node import Node
from causal_reasoning.linear_algorithm.linear_constraints import generate_constraints
from causal_reasoning.linear_algorithm.obj_function_generator import (
    ObjFunctionGenerator,
)


def build_linear_problem(
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
    objFG.find_linear_good_set()
    mechanisms = objFG.get_mechanisms_pruned()

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
            logger.debug(f"{decisionMatrix[i][j]} ")
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

    logger.info(
        f"Causal query: P({target.label}={target.value}|do({intervention.label}={intervention.value}))"
    )
    logger.info(f"Bounds: {lowerBound} <= P <= {upperBound}")


def build_bi_linear_problem(
    graph: Graph,
    df: pd.DataFrame,
    interventions: tuple[Node],
    target: Node,
):
    multiObjFG = DoubleObjFunctionGenerator(
        graph=graph,
        dataFrame=df,
        interventions=(interventions[0], interventions[1]),
        target=target,
    )

    multiObjFG.find_linear_good_set()
    mechanisms_1, mechanisms_2 = multiObjFG.get_mechanisms_pruned()


    interventionLatentParent_1 = multiObjFG.intervention_1.latentParent
    cComponentEndogenous_1 = interventionLatentParent_1.children
    consideredEndogenousNodes_1 = list(
        (set(cComponentEndogenous_1) & set(multiObjFG.debugOrder)) | {multiObjFG.intervention_1}
    )
    probs_1, decisionMatrix_1 = generate_constraints(
        data=df,
        dag=multiObjFG.graph,
        unob=interventionLatentParent_1,
        consideredCcomp=consideredEndogenousNodes_1,
        mechanisms=mechanisms_1,
    )

    interventionLatentParent_2 = multiObjFG.intervention_2.latentParent
    cComponentEndogenous_2 = interventionLatentParent_2.children
    consideredEndogenousNodes_2 = list(
        (set(cComponentEndogenous_2) & set(multiObjFG.debugOrder)) | {multiObjFG.intervention_2}
    )
    probs_2, decisionMatrix_2 = generate_constraints(
        data=df,
        dag=multiObjFG.graph,
        unob=interventionLatentParent_2,
        consideredCcomp=consideredEndogenousNodes_2,
        mechanisms=mechanisms_2,
    )

    # Dupla intervenção pode virar binomial.
    # c_i u_i ==> c_i_j * u_i * w_j (para intervenções em c components diferentes)
    objFunctionCoefficients: list[list[float]] = multiObjFG.build_objective_function(mechanisms_1, mechanisms_2)

    model = gp.Model("linear")

    number_of_vars_1 = len(decisionMatrix_1[0])
    number_of_vars_2 = len(decisionMatrix_2[0])

    vars = model.addVars(number_of_vars_1 + number_of_vars_2, obj=1, name="Variables")

    constrs_1 = model.addConstrs(
        (gp.quicksum(decisionMatrix_1[i][j] * vars[j] for j in range(number_of_vars_1)) == probs_1[i]
        for i in range(len(probs_1))),
        name="Intervention_1_Related_Constraints"
    )

    constrs_2 = model.addConstrs(
        (gp.quicksum(decisionMatrix_2[i][j] * vars[j + number_of_vars_1] for j in range(number_of_vars_2)) == probs_2[i]
        for i in range(len(probs_2))),
        name="Intervention_2_Related_Constraints"
    )

    model.setObjective(
        gp.quicksum(objFunctionCoefficients[i][j] * vars[i] * vars[j] for i in range(number_of_vars_1) for j in range(number_of_vars_2)),
    )

    model.params.outputFlag = 0

    model.modelSense = gp.GRB.MINIMIZE
    model.optimize()
    # duals_1 = model.getAttr("pi", constrs_1)
    # duals_2 = model.getAttr("pi", constrs_2)
    # logger.info(f"duals: {duals_1}, {duals_2}")

    if model.Status == gp.GRB.OPTIMAL: # OPTIMAL
            lower = model.objVal
            logger.info(f"Minimal solution found! -- MIN Query: {lower}")
    else:
        logger.info(f"Minimal solution not found. Gurobi status code: {model.Status}")
        lower = None


    model.modelSense = gp.GRB.MAXIMIZE
    model.optimize()
    # duals_1 = model.getAttr("pi", constrs_1)
    # duals_2 = model.getAttr("pi", constrs_2)
    # logger.info(f"duals: {duals_1}, {duals_2}")
    if model.Status == gp.GRB.OPTIMAL: # OPTIMAL
            upper = model.objVal
            logger.info(f"Maximal solution found! -- MAX Query: {upper}")
    else:
        logger.info(f"Maximal solution not found. Gurobi status code: {model.Status}")
        upper = None
    
    logger.info(f"Query interval = [{lower, upper}]")
