from typing import List, Tuple
import logging
from abc import ABC, abstractmethod

import gurobipy as gp
from scipy.optimize import linprog

logger = logging.getLogger(__name__)

from causal_reasoning.utils._enum import OptimizationDirection, OptimizersLabels


class Optimizer(ABC):
    """
    Abstract base class for linear programming optimizers.
    """

    def __init__(
        self,
        probs: List[float],
        decision_matrix: List[List[int]],
        obj_function_coefficients: List[float],
    ) -> None:
        if len(decision_matrix) != len(probs):
            raise ValueError("decision_matrix row count must match length of probs")
        if len(decision_matrix[0]) != len(obj_function_coefficients):
            raise ValueError("Number of columns in decision_matrix must match length of obj_function_coefficients")
        self.probs = probs
        self.decision_matrix = decision_matrix
        self.obj_function_coefficients = obj_function_coefficients

    @property
    def num_vars(self) -> int:
        return len(self.decision_matrix[0])

    def template_method(self) -> Tuple[str, str]:
        self.setup()
        return self.compute_bounds()

    @abstractmethod
    def setup(self) -> None:
        pass

    def compute_bounds(self) -> Tuple[str, str]:
        lowerBound = self.run_optimizer(OptimizationDirection.MINIMIZE)
        upperBound = self.run_optimizer(OptimizationDirection.MAXIMIZE)
        return lowerBound, upperBound

    @abstractmethod
    def run_optimizer(self, is_minimization: bool) -> str:
        pass

    def get_standardized_problem(self):
        A_eq = self.decision_matrix
        b_eq = self.probs
        obj_coeffs = self.obj_function_coefficients
        intervals = [(0, 1)] * self.num_vars
        return A_eq, b_eq, obj_coeffs, intervals


class GurobiOptimizer(Optimizer):
    def __init__(
        self,
        probs: List[float],
        decision_matrix: List[List[int]],
        obj_function_coefficients: List[float],
    ) -> None:
        super().__init__(probs, decision_matrix, obj_function_coefficients)

        self.model = gp.Model("linear")
        self.vars = None
        self.constrs = None

    def setup(
        self,
        model_sense,
    ):
        A_eq, b_eq, obj_coeffs, _ = self.get_standardized_problem()
        self.vars = self.model.addVars(self.num_vars, obj=1, name="Variables")

        self.constrs = self.model.addConstrs(
            (
                gp.quicksum(
                    A_eq[i][j] * self.vars[j] for j in range(self.num_vars)
                )
                == b_eq[i]
                for i in range(len(b_eq))
            ),
            name="Constraints",
        )

        self.model.model_sense = model_sense

        self.model.setObjective(
            gp.quicksum(
                obj_coeffs[i] * self.vars[i]
                for i in range(len(obj_coeffs))
            )
        )

        self.model.params.outputFlag = 0
        self.model.update()

    def run_optimizer(self, direction: OptimizationDirection) -> str:
        if direction == OptimizationDirection.MINIMIZE:
            model_sense = gp.GRB.MINIMIZE
            msg = "Minimal"
        else:
            model_sense = gp.GRB.MAXIMIZE
            msg = "Maximal"

        self.setup(model_sense)
        self.model.optimize()

        if self.model.Status == gp.GRB.OPTIMAL:
            bound = self.model.objVal
            logger.info(f"{msg} solution found!\nMIN Query: {bound}")
            return str(bound)

        logger.info(
            f"{msg} solution not found. Gurobi status code: {self.model.Status}"
        )
        return "None"


class ScipyOptimizer(Optimizer):
    def __init__(
        self,
        probs: List[float],
        decision_matrix: List[List[int]],
        obj_function_coefficients: List[float],
    ) -> None:
        super().__init__(probs, decision_matrix, obj_function_coefficients)

    def run_optimizer(self, direction: OptimizationDirection) -> str:
        A_eq, b_eq, obj_coeffs, intervals = self.get_standardized_problem()
        coef = 1
        msg = "Maximal"
        if direction == OptimizationDirection.MINIMIZE:
            coef = -1
            msg = "Minimal"

        result = linprog(
            c=[coef * x for x in obj_coeffs],
            A_ub=None,
            b_ub=None,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=intervals,
            method="highs"
        )

        logger.info(f"{msg} Success: {result.success}")
        logger.info(f"{msg} Status: {result.status}")
        logger.info(f"{msg} Message: {result.message}")

        if result is None:
            logger.info(f"{msg} Sol is None")
            return "None"
        elif result.fun is None:
            logger.info(f"{msg}Sol.fun is None")
            return "None"
        else:
            if direction:
                return str(result.fun)
            else:
                return str(-result.fun)


def choose_optimizer(
    optimizer_label: str,
    probs: List[float],
    decision_matrix: List[List[int]],
    obj_function_coefficients: List[float],
) -> Tuple[str, str]:

    if optimizer_label == OptimizersLabels.GUROBI.value:
        return GurobiOptimizer(probs, decision_matrix, obj_function_coefficients)

    if optimizer_label == OptimizersLabels.SCIPY.value:
        return ScipyOptimizer(probs, decision_matrix, obj_function_coefficients)

    raise Exception(f"Optimizer {optimizer_label} not found.")


def compute_bounds(optimizer: Optimizer) -> Tuple[str, str]:
    return optimizer.template_method()
