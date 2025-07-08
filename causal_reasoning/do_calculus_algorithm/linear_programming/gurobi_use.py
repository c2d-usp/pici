import numpy as np
import gurobipy as gp
import pandas as pd

import logging

logger = logging.getLogger(__name__)

from causal_reasoning.graph.graph import Graph
from causal_reasoning.graph.node import Node
from causal_reasoning.do_calculus_algorithm.linear_programming.linear_constraints import (
    generate_constraints,
)
from causal_reasoning.do_calculus_algorithm.linear_programming.obj_function_generator import (
    ObjFunctionGenerator,
)


class MasterProblem:
    def __init__(self):
        self.model = gp.Model("linear")
        self.vars = None
        self.constrs = None

    def setup(
        self,
        probs,
        decisionMatrix,
        objFunctionCoefficients: list[float],
        modelSense: int = 1,
    ):
        num_vars = len(decisionMatrix[0])
        self.vars = self.model.addVars(num_vars, obj=1, name="Variables")

        self.constrs = self.model.addConstrs(
            (
                gp.quicksum(
                    decisionMatrix[i][j] * self.vars[j] for j in range(num_vars)
                )
                == probs[i]
                for i in range(len(probs))
            ),
            name="Constraints",
        )

        if modelSense == 1:
            self.model.modelSense = gp.GRB.MINIMIZE
        else:
            self.model.modelSense = gp.GRB.MAXIMIZE
        self.model.setObjective(
            gp.quicksum(
                objFunctionCoefficients[i] * self.vars[i]
                for i in range(len(objFunctionCoefficients))
            )
        )

        # Turning off output because of the iterative procedure
        self.model.params.outputFlag = 0
        self.model.update()

    def update(self, pattern, index):
        new_col = gp.Column(coeffs=pattern, constrs=self.constrs.values())
        self.vars[index] = self.model.addVar(
            obj=1, column=new_col, name=f"Pattern[{index}]"
        )
        self.model.update()


def gurobi_build_linear_problem(
    graph: Graph,
    df: pd.DataFrame,
    intervention: Node,
    target: Node,
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

    probs, decisionMatrix = generate_constraints(
        data=df,
        dag=objFG.graph,
        unob=interventionLatentParent,
        consideredCcomp=consideredEndogenousNodes,
        mechanisms=mechanisms,
    )
    objFunctionCoefficients: list[float] = objFG.build_objective_function(mechanisms)

    master = MasterProblem()
    modelSenseMin = 1
    master.setup(probs, decisionMatrix, objFunctionCoefficients, modelSenseMin)

    master.model.optimize()
    # duals = master.model.getAttr("pi", master.constrs)
    # logger.info(f"duals: {duals}")
    if master.model.Status == gp.GRB.OPTIMAL:  # OPTIMAL
        lower = master.model.objVal
        logger.info(f"Minimal solution found!\nMIN Query: {lower}")
    else:
        logger.info(
            f"Minimal solution not found. Gurobi status code: {master.model.Status}"
        )
        lower = None
    modelSenseMax = -1
    master.setup(probs, decisionMatrix, objFunctionCoefficients, modelSenseMax)

    master.model.optimize()
    # duals = master.model.getAttr("pi", master.constrs)
    # logger.info(f"duals: {duals}")
    if master.model.Status == gp.GRB.OPTIMAL:  # OPTIMAL
        upper = master.model.objVal
        logger.info(f"Maximal solution found!\nMAX Query: {upper}")
    else:
        logger.info(
            f"Maximal solution not found. Gurobi status code: {master.model.Status}"
        )
        upper = None

    logger.info(f"Query interval = [{lower, upper}]")
    return str(lower), str(upper)
