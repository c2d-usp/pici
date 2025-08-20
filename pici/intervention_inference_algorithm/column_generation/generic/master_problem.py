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

    def setup(
        self,
        columns_base: list[list[int]],
        constraints_empirical_probabilities: list[float],
    ):
        """
        Initializes the master problem with base columns and empirical probability constraints.

        Args:
            columns_base (list[list[int]]): The base columns is an identity matrix for the initial variables.
            constraints_empirical_probabilities (list[float]): The right-hand side values for the empirical probability constraints.

        This method creates variables for each base column, sets up the constraints so that the
        linear combination of columns matches the empirical probabilities, and configures the model
        for minimization. Gurobi output is suppressed for iterative procedures.
        """
        num_columns_base = len(columns_base)
        self.vars = self.model.addVars(num_columns_base, obj=BIG_M, name="BaseColumns")
        self.constrs = self.model.addConstrs(
            (
                gp.quicksum(
                    columns_base[column_id][realization_id] * self.vars[column_id]
                    for column_id in range(num_columns_base)
                )
                == constraints_empirical_probabilities[realization_id]
                for realization_id in range(len(constraints_empirical_probabilities))
            ),
            name="EmpiricalRestrictions",
        )
        self.model.modelSense = GRB.MINIMIZE
        # Turning off output because of the iterative procedure
        self.model.params.outputFlag = 0
        # self.model.setParam('FeasibilityTol', 1e-9)
        self.model.update()

    def update(
        self, new_column: list[float], index: int, obj_coeff: list[float], minimun: bool
    ):
        """
        Adds a new column (variable) to the constraints in the master problem and updates the model.

        Args:
            new_column (list[float]): The coefficients of the new variable for each constraint.
            index (int): The index at which to add the new variable.
            obj_coeff (list[float]): The objective coefficient(s) in the objective function for the new variable.
            minimun (bool): If True, the objective is minimized; if False, the coefficient is negated for maximization.

        This method constructs a new Gurobi variable with the specified column and objective coefficient,
        adds it to the model, and updates the model structure.
        """
        new_col = gp.Column(
            coeffs=new_column, constrs=self.constrs.values()
        )  # Includes the new variable in the constraints
        logger.debug(f"Obj coeff: {obj_coeff}")
        if minimun:
            self.vars[index] = self.model.addVar(
                obj=obj_coeff,
                column=new_col,  # Adds the new variable
                name=f"Variable[{index}]",
            )
        else:
            self.vars[index] = self.model.addVar(
                obj=-obj_coeff,
                column=new_col,  # Adds the new variable
                name=f"Variable[{index}]",
            )
        self.model.update()
