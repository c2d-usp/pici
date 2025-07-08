from abc import ABC, abstractmethod
import gurobipy as gp
from scipy.optimize import linprog

import logging

logger = logging.getLogger(__name__)

from causal_reasoning.utils._enum import OptimizersLabels


class Optimizer(ABC):
    def __init__(self,
        probs: list[float],
        decisionMatrix: list[list[int]],
        objFunctionCoefficients: list[float],
    ) -> None:
        self.probs = probs
        self.decisionMatrix = decisionMatrix
        self.objFunctionCoefficients = objFunctionCoefficients
    
    def template_method(self) -> tuple[str, str]:
        self.setup()
        return self.compute_bounds()
    
    @abstractmethod
    def setup(self) -> None:
        pass

    def compute_bounds(self) -> tuple[str, str]:
        lowerBound = self.run_optimizer(isMinimization=1)
        upperBound = self.run_optimizer(isMinimization=0)
        return lowerBound, upperBound

    def run_optimizer(self) -> str:
        return "None"


class GurobiOptimizer(Optimizer):
    def __init__(self, probs: list[float], decisionMatrix: list[list[int]], objFunctionCoefficients: list[float]) -> None:
        super().__init__(probs, decisionMatrix, objFunctionCoefficients)

        self.model = gp.Model("linear")
        self.vars = None
        self.constrs = None

    def setup(
        self,
        modelSense: int = 1,
    ):
        num_vars = len(self.decisionMatrix[0])
        self.vars = self.model.addVars(num_vars, obj=1, name="Variables")

        self.constrs = self.model.addConstrs(
            (
                gp.quicksum(
                    self.decisionMatrix[i][j] * self.vars[j] for j in range(num_vars)
                )
                == self.probs[i]
                for i in range(len(self.probs))
            ),
            name="Constraints",
        )

        if modelSense == 1:
            self.model.modelSense = gp.GRB.MINIMIZE
        else:
            self.model.modelSense = gp.GRB.MAXIMIZE

        self.model.setObjective(
            gp.quicksum(
                self.objFunctionCoefficients[i] * self.vars[i]
                for i in range(len(self.objFunctionCoefficients))
            )
        )

        self.model.params.outputFlag = 0
        self.model.update()

    def run_optimizer(self, isMinimization: bool) -> str:
        if isMinimization:
            modelSense = 1
            msg = "Minimal"
        else:
            modelSense = -1
            msg = "Maximal"

        self.setup(modelSense)
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
    def __init__(self, probs: list[float], decisionMatrix: list[list[int]], objFunctionCoefficients: list[float]) -> None:
        super().__init__(probs, decisionMatrix, objFunctionCoefficients)
    
    def setup(self):
        self.intervals = [(0, 1) for _ in range(len(self.decisionMatrix[0]))]

    def run_optimizer(self, isMinimization: bool) -> str:
        if isMinimization:
            coef = -1
            msg = "Lower Bound"
        else:
            coef = 1
            msg = "Upper Bound"

        boundSol = linprog(
            c=[coef * x for x in self.objFunctionCoefficients],
            A_ub=None,
            b_ub=None,
            A_eq=self.decisionMatrix,
            b_eq=self.probs,
            method="highs",
            bounds=self.intervals,
        )
        logger.info(f"{msg} Success: {boundSol.success}")
        logger.info(f"{msg} Status: {boundSol.status}")
        logger.info(f"{msg} Message: {boundSol.message}")

        if boundSol is None:
            logger.info(f"{msg} Sol is None")
            return "None"
        elif boundSol.fun is None:
            logger.info(f"{msg}Sol.fun is None")
            return "None"
        else:
            if isMinimization:
                return str(boundSol.fun)
            else:
                return str(-boundSol.fun)


def choose_optimizer(
    optimizer_label: str,
    probs: list[float],
    decisionMatrix: list[list[int]],
    objFunctionCoefficients: list[float]
) -> tuple[str, str]:

    if optimizer_label == OptimizersLabels.GUROBI.value:
        return GurobiOptimizer(probs, decisionMatrix, objFunctionCoefficients)

    if optimizer_label == OptimizersLabels.SCIPY.value:
        return ScipyOptimizer(probs, decisionMatrix, objFunctionCoefficients)

    raise Exception(f"Optimizer {optimizer_label} not found.")
    

def compute_bounds(optimizer: Optimizer) -> tuple[str, str]:
    return optimizer.template_method()

