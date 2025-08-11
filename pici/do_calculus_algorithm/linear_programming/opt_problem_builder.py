import logging

import gurobipy as gp
import pandas as pd

from pici.do_calculus_algorithm.linear_programming.double_intervention_obj_func_gen import (
    DoubleInterventionObjFunctionGenerator,
)

logger = logging.getLogger(__name__)

from pici.do_calculus_algorithm.linear_programming.linear_constraints import (
    generate_constraints,
)
from pici.do_calculus_algorithm.linear_programming.obj_function_generator import (
    ObjFunctionGenerator,
)
from pici.do_calculus_algorithm.linear_programming.optimizers import (
    Optimizer,
    choose_optimizer,
    compute_bounds,
)
from pici.graph.graph import Graph
from pici.graph.node import Node
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
    objFG.find_linear_good_set()
    mechanisms = objFG.get_mechanisms_pruned()

    interventionLatentParent = objFG.intervention.latentParent
    cComponentEndogenous = interventionLatentParent.children
    consideredEndogenousNodes = list(
        (set(cComponentEndogenous) & set(objFG.debugOrder)) | {objFG.intervention}
    )

    probs, decision_matrix = generate_constraints(
        data=df,
        dag=objFG.graph,
        unob=interventionLatentParent,
        consideredCcomp=consideredEndogenousNodes,
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


def build_bi_linear_problem(
    graph: Graph,
    df: pd.DataFrame,
    interventions: tuple[Node],
    target: Node,
) -> tuple[str, str]:
    multiObjFG = DoubleInterventionObjFunctionGenerator(
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
        (set(cComponentEndogenous_1) & set(multiObjFG.debugOrder))
        | {multiObjFG.intervention_1}
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
        (set(cComponentEndogenous_2) & set(multiObjFG.debugOrder))
        | {multiObjFG.intervention_2}
    )
    probs_2, decisionMatrix_2 = generate_constraints(
        data=df,
        dag=multiObjFG.graph,
        unob=interventionLatentParent_2,
        consideredCcomp=consideredEndogenousNodes_2,
        mechanisms=mechanisms_2,
    )

    obj_function_coefficients: list[list[float]] = multiObjFG.build_objective_function(
        mechanisms_1, mechanisms_2
    )

    model = gp.Model("linear")

    number_of_vars_1 = len(decisionMatrix_1[0])
    number_of_vars_2 = len(decisionMatrix_2[0])

    vars = model.addVars(number_of_vars_1 + number_of_vars_2, obj=1, name="Variables")

    constrs_1 = model.addConstrs(
        (
            gp.quicksum(
                decisionMatrix_1[i][j] * vars[j] for j in range(number_of_vars_1)
            )
            == probs_1[i]
            for i in range(len(probs_1))
        ),
        name="Intervention_1_Related_Constraints",
    )

    constrs_2 = model.addConstrs(
        (
            gp.quicksum(
                decisionMatrix_2[i][j] * vars[j + number_of_vars_1]
                for j in range(number_of_vars_2)
            )
            == probs_2[i]
            for i in range(len(probs_2))
        ),
        name="Intervention_2_Related_Constraints",
    )

    model.setObjective(
        gp.quicksum(
            obj_function_coefficients[i][j] * vars[i] * vars[j]
            for i in range(number_of_vars_1)
            for j in range(number_of_vars_2)
        ),
    )

    model.params.outputFlag = 0

    model.modelSense = gp.GRB.MINIMIZE
    model.optimize()
    # duals_1 = model.getAttr("pi", constrs_1)
    # duals_2 = model.getAttr("pi", constrs_2)
    # logger.info(f"duals: {duals_1}, {duals_2}")

    if model.Status == gp.GRB.OPTIMAL:  # OPTIMAL
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
    if model.Status == gp.GRB.OPTIMAL:  # OPTIMAL
        upper = model.objVal
        logger.info(f"Maximal solution found! -- MAX Query: {upper}")
    else:
        logger.info(f"Maximal solution not found. Gurobi status code: {model.Status}")
        upper = None

    logger.info(f"Query interval = [{lower, upper}]")
    return str(lower), str(upper)
