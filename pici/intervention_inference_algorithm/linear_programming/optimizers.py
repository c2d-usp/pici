from abc import ABC, abstractmethod
import logging
from typing import List, Tuple

from gurobipy import GRB, Model, quicksum
from scipy.optimize import linprog

logger = logging.getLogger(__name__)

from pici.utils._enum import (
    GurobiParameters,
    OptimizationDirection,
    OptimizersLabels,
)


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
            raise ValueError(
                "Number of columns in decision_matrix must match length of obj_function_coefficients"
            )
        self.probs = probs
        self.decision_matrix = decision_matrix
        self.obj_function_coefficients = obj_function_coefficients

    @property
    def num_vars(self) -> int:
        return len(self.decision_matrix[0])

    def template_method(self) -> Tuple[str, str]:
        return self.compute_bounds()

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

        self.model = Model("linear")
        self.vars = None
        self.constrs = None

    def run_optimizer(self, direction: OptimizationDirection) -> str:
        if direction == OptimizationDirection.MINIMIZE:
            model_sense = GRB.MINIMIZE
            msg = "Minimal"
        else:
            model_sense = GRB.MAXIMIZE
            msg = "Maximal"

        A_eq, b_eq, obj_coeffs, _ = self.get_standardized_problem()
        self.setup_variables_and_constraints(A_eq, b_eq)
        self.configure_objective(obj_coeffs)
        self.configure_solver_params(model_sense)
        self.model.optimize()

        if self.model.Status == GRB.OPTIMAL:
            bound = self.model.objVal
            logger.info(f"{msg} solution found! {msg} Query: {bound}")
            return str(bound)

        logger.info(
            f"{msg} solution not found. Gurobi status code: {self.model.Status}"
        )
        return "None"

    def setup_variables_and_constraints(self, A_eq, b_eq):
        self.vars = self.model.addVars(
            self.num_vars,
            obj=GurobiParameters.DefaultObjectiveCoefficients.value,
            name="Variables",
        )

        self.constrs = self.model.addConstrs(
            (
                quicksum(A_eq[i][j] * self.vars[j] for j in range(self.num_vars))
                == b_eq[i]
                for i in range(len(b_eq))
            ),
            name="Constraints",
        )

    def configure_objective(self, obj_coeffs):
        self.model.setObjective(
            quicksum(obj_coeffs[i] * self.vars[i] for i in range(len(obj_coeffs)))
        )

    def configure_solver_params(self, model_sense):
        self.model.model_sense = model_sense
        self.model.params.outputFlag = GurobiParameters.OUTPUT_SUPRESSED.value
        self.model.update()


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
        sign = 1
        msg = "Maximal"
        if direction == OptimizationDirection.MINIMIZE:
            sign = -1
            msg = "Minimal"

        result = linprog(
            c=[sign * x for x in obj_coeffs],
            A_ub=None,
            b_ub=None,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=intervals,
            method="highs",
        )

        logger.info(f"{msg} Success: {result.success}")
        logger.info(f"{msg} Status: {result.status}")
        logger.info(f"{msg} Message: {result.message}")

        if result is None or result.fun is None:
            logger.info(f"{msg} query is None")
            return "None"

        if direction == OptimizationDirection.MINIMIZE:
            return str(result.fun)

        return str(-result.fun)


def choose_optimizer(
    optimizer_label: str,
    probs: List[float],
    decision_matrix: List[List[int]],
    obj_function_coefficients: List[float],
) -> Optimizer:

    if optimizer_label == OptimizersLabels.GUROBI.value:
        return GurobiOptimizer(probs, decision_matrix, obj_function_coefficients)

    if optimizer_label == OptimizersLabels.SCIPY.value:
        return ScipyOptimizer(probs, decision_matrix, obj_function_coefficients)

    raise Exception(f"Optimizer {optimizer_label} not found.")


def compute_bounds(optimizer: Optimizer) -> Tuple[str, str]:
    return optimizer.template_method()
