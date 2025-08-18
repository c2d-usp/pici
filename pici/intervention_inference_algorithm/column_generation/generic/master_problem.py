import logging

import gurobipy as gp
from gurobipy import GRB

logger = logging.getLogger(__name__)


import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

BIG_M = 1e4
DBG = False
MAX_ITERACTIONS_ALLOWED = 2000


class MasterProblem:
    def __init__(self):
        self.model = gp.Model("master")
        self.vars = None
        self.constrs = None

    def setup(self, columns_base: list[list[int]], empiricalProbabilities: list[float]):
        num_columns_base = len(columns_base)
        self.vars = self.model.addVars(num_columns_base, obj=BIG_M, name="BaseColumns")
        self.constrs = self.model.addConstrs(
            (
                gp.quicksum(
                    columns_base[column_id][realization_id] * self.vars[column_id]
                    for column_id in range(num_columns_base)
                )
                == empiricalProbabilities[realization_id]
                for realization_id in range(len(empiricalProbabilities))
            ),
            name="EmpiricalRestrictions",
        )
        self.model.modelSense = GRB.MINIMIZE
        # Turning off output because of the iterative procedure
        self.model.params.outputFlag = 0
        # self.model.setParam('FeasibilityTol', 1e-9)
        self.model.update()

    def update(
        self, newColumn: list[float], index: int, objCoeff: list[float], minimun: bool
    ):
        new_col = gp.Column(
            coeffs=newColumn, constrs=self.constrs.values()
        )  # Includes the new variable in the constraints
        logger.debug(f"Obj coeff: {objCoeff}")
        if minimun:
            self.vars[index] = self.model.addVar(
                obj=objCoeff,
                column=new_col,  # Adds the new variable
                name=f"Variable[{index}]",
            )
        else:
            self.vars[index] = self.model.addVar(
                obj=-objCoeff,
                column=new_col,  # Adds the new variable
                name=f"Variable[{index}]",
            )
        self.model.update()

