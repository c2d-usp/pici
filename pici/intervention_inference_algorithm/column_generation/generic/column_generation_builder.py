import copy
import logging

import gurobipy as gp
from gurobipy import GRB

from pici.graph.graph import Graph
from pici.graph.node import Node
from pici.intervention_inference_algorithm.column_generation.generic import bits
from pici.intervention_inference_algorithm.column_generation.generic.master_problem import MasterProblem
from pici.intervention_inference_algorithm.column_generation.generic.subproblem import SubProblem
from pici.intervention_inference_algorithm.linear_programming.linear_constraints import find_c_component_and_tail_set, get_c_component_in_reverse_topological_order, get_symbolical_constraints_probabilities_and_wc
from pici.intervention_inference_algorithm.linear_programming.obj_function_generator import ObjFunctionGenerator
from pici.utils.scalable_graphs_helper import get_scalable_dataframe

logger = logging.getLogger(__name__)


import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pici.intervention_inference_algorithm.column_generation.scalable_problem_init import (
    InitScalable,
)

BIG_M = 1e4
DBG = False
MAX_ITERACTIONS_ALLOWED = 2000


class ColumnGenerationProblemBuilder:
    def __init__(
        self,
        dataFrame,
        empiricalProbabilities: list[float],
        parametric_columns: dict[str, tuple[list[int]]],
        betaVarsCost: list[float],
        betaVarsBitsX0: list[tuple[str]],
        betaVarsBitsX1: list[tuple[str]],

        dag: Graph,
        intervention: Node,
        target: Node,

        minimum: bool,
    ):
        
        self.intervention = intervention
        self.target = target
        self.dataFrame = dataFrame

        objective_function = ObjFunctionGenerator(
            graph=dag,
            dataFrame=dataFrame,
            intervention=intervention,
            target=target,
        )

        interventionLatentParent = objective_function.intervention.latentParent
        cComponentEndogenous = interventionLatentParent.children
        consideredCcomp = list(
            (set(cComponentEndogenous) & set(objective_function.consideredGraphNodes)) | {objective_function.intervention}
        )

        topoOrder: list[Node] = dag.topologicalOrder

        cCompOrder = get_c_component_in_reverse_topological_order(
            topoOrder=topoOrder, unob=intervention.latentParent, consideredCcomp=consideredCcomp
        )
        c_component_and_tail: list[Node] = find_c_component_and_tail_set(intervention.latentParent, cCompOrder)
        symbolical_constraints_probabilities, W = get_symbolical_constraints_probabilities_and_wc(cCompOrder=cCompOrder, c_component_and_tail=c_component_and_tail, topoOrder=topoOrder)

        self.amountNonTrivialRestrictions = len(W)

        auxDict = {}
        for i in range(self.amountNonTrivialRestrictions):
            auxDict[i] = BIG_M
        self.duals = auxDict.copy()

        self.symbolic_objective_function_probabilites: list[tuple] = objective_function.generate_symbolic_objective_function_probabilities()

        self.symbolic_decision_function: dict[tuple, int] = objective_function.generate_symbolic_decision_function()

        self.bits_list: list[int] = bits.generate_optimization_problem_bit_list(intervention)

        # TODO: Empirical não é symbolical_objective_function_probabilities 
        self.empiricalProbabilities: list[float] = empiricalProbabilities

        self.columns_base = None

        # vvvvvv Bloco antigo vvvvvvvvvv
        N = 1
        M = 1
        self.amountBitsPerCluster = 1 << (M + 1)
        self.amountBetaVarsPerX = 1 << (M + N)

        # Order parametric_columns (XA1A2..AnB1...Bm)
        self.parametric_columns: dict[str, tuple[list[int]]] = parametric_columns
        self.betaVarsBitsX0 = betaVarsBitsX0
        self.betaVarsBitsX1 = betaVarsBitsX1
        self.betaVarsCost = betaVarsCost
        self.minimum = minimum

        self.solution = {}
        self.master = MasterProblem()
        self.subproblem = SubProblem(N=N, M=M)

    def _initialize_column_base(self):
        # Initialize big-M problem with the identity block of size
        # equal to the amount of restrictions.
        columns_base: list[list[int]] = []
        for index in range(self.amountNonTrivialRestrictions + 1):
            new_column = [0] * (self.amountNonTrivialRestrictions + 1)
            new_column[index] = 1
            columns_base.append(new_column)
        self.columns_base = columns_base

    def _generate_patterns(self):
        self._initialize_column_base()
        ####
        self.master.setup(self.columns_base, self.empiricalProbabilities)
        self.subproblem.setup(
            amountBitsPerCluster=self.amountBitsPerCluster,
            amountBetaVarsPerX=self.amountBetaVarsPerX,
            duals=self.duals,
            amountNonTrivialRestrictions=self.amountNonTrivialRestrictions,
            betaVarsCost=self.betaVarsCost,
            parametric_column=self.parametric_columns,
            betaVarsBitsX0=self.betaVarsBitsX0,
            betaVarsBitsX1=self.betaVarsBitsX1,
            N=self.N,
            M=self.M,
            interventionValue=self.intervention.value,
            minimum=self.minimum,
        )

        counter = 0
        while True:
            self.master.model.optimize()
            if self.master.model.Status == gp.GRB.OPTIMAL:  # OPTIMAL
                b = self.master.model.objVal
                logger.info(f"--------->> Master solution found: {b}")
            elif self.master.model.Status == gp.GRB.USER_OBJ_LIMIT:
                b = self.master.model.objVal
                logger.info(
                    f"--------->> BIG_M Limit reached! Master solution found: {b}"
                )
            else:
                logger.error(
                    f"--------->>  Master solution not found. Gurobi status code: {self.master.model.Status}"
                )
            self.duals = self.master.model.getAttr("pi", self.master.constrs)
            logger.debug(f"Master Duals: {self.duals}")
            # self.master.model.write(f"master_{counter}.lp")
            self.subproblem.update(self.duals)
            self.subproblem.model.optimize()
            if self.subproblem.model.Status == gp.GRB.OPTIMAL:  # OPTIMAL
                b = self.subproblem.model.objVal
                logger.info(f"--------->> Subproblem solution found!: {b}")
            elif self.subproblem.model.Status == gp.GRB.USER_OBJ_LIMIT:
                b = self.subproblem.model.objVal
                logger.info(
                    f"--------->> BIG_M Limit reached! Subproblem solution found: {b}"
                )
            else:
                logger.error(
                    f"--------->>  Subproblem solution not found. Gurobi status code: {self.subproblem.model.Status}"
                )
            # self.subproblem.model.write(f"subproblem_{counter}.lp")

            reduced_cost = self.subproblem.model.objVal
            logger.debug(f"Reduced Cost: {reduced_cost}")
            if reduced_cost >= 0:
                break

            newColumn: list[int] = []
            for index in range(len(self.subproblem.bitsParametric)):
                newColumn.append(self.subproblem.bitsParametric[index].X)

            newColumn.append(
                1
            )  # For the equation sum(pi) = 1. This restriction is used in the MASTER problem.
            logger.debug(f"New Column: {newColumn}")

            objCoeff: float = 0.0
            for betaIndex in range(self.amountBetaVarsPerX):
                if self.intervention.value == 0:
                    objCoeff += (
                        self.betaVarsCost[betaIndex]
                        * self.subproblem.beta_varsX0[betaIndex].X
                    )
                else:
                    objCoeff += (
                        self.betaVarsCost[betaIndex]
                        * self.subproblem.beta_varsX1[betaIndex].X
                    )

            self.master.update(
                newColumn=newColumn,
                index=len(self.columns_base),
                objCoeff=objCoeff,
                minimun=self.minimum,
            )
            self.columns_base.append(newColumn)
            counter += 1
            if counter >= MAX_ITERACTIONS_ALLOWED:
                raise TimeoutError(
                    f"Too many iterations (MAX:{MAX_ITERACTIONS_ALLOWED})"
                )
            logger.info(f"Iteration Number = {counter}")

        return counter

    def solve(self, method=1, presolve=-1, numeric_focus=-1, opt_tol=-1, fea_tol=-1):
        """
        Gurobi does not support branch-and-price, as this requires to add columns
        at local nodes of the search tree. A heuristic is used instead, where the
        integrality constraints for the variables of the final root LP relaxation
        are installed and the resulting MIP is solved. Note that the optimal
        solution could be overlooked, as additional columns are not generated at
        the local nodes of the search tree.
        """
        self.master.model.params.Method = method
        self.subproblem.model.params.Method = method

        if presolve != -1:
            self.master.model.Params.Presolve = presolve
            self.subproblem.model.Params.Presolve = presolve
        if numeric_focus != -1:
            self.master.model.Params.NumericFocus = numeric_focus
            self.subproblem.model.Params.NumericFocus = numeric_focus

        if opt_tol != -1:
            self.master.model.Params.OptimalityTol = opt_tol
            self.subproblem.model.Params.OptimalityTol = opt_tol

        if fea_tol != -1:
            self.master.model.Params.FeasibilityTol = fea_tol
            self.subproblem.model.Params.FeasibilityTol = fea_tol

        numberIterations = self._generate_patterns()
        self.master.model.setAttr("vType", self.master.vars, GRB.CONTINUOUS)
        self.master.model.optimize()
        self.master.model.write("model.lp")
        self.master.model.write("model.mps")
        bound = self.master.model.ObjVal
        itBound = numberIterations
        return bound, itBound


def buildScalarProblem(
    M: int, N: int, interventionValue: int, targetValue: int, df, minimum: bool
):
    # Calculate the empirical probs (RHS of the restrictions, so b in Ax=b)
    empiricalProbabilities: list[float] = InitScalable.calculateEmpiricals(
        N=N, M=M, df=df, DBG=DBG
    )
    # Auxiliary Gamma U variables (beta): calculate the obj coeff in the subproblem and the relation to the bit variables that compose them
    betaVarsCoeffObjSubproblem: list[float] = []
    betaVarsBitsX0, betaVarsCoeffObjSubproblemX0 = (
        InitScalable.defineGammaUAuxiliaryVariables(
            M=M, N=N, df=df, targetValue=targetValue, XValue=0, DBG=DBG
        )
    )
    betaVarsBitsX1, betaVarsCoeffObjSubproblemX1 = (
        InitScalable.defineGammaUAuxiliaryVariables(
            M=M, N=N, df=df, targetValue=targetValue, XValue=1, DBG=DBG
        )
    )
    if interventionValue == 1:
        betaVarsCoeffObjSubproblem = copy.deepcopy(betaVarsCoeffObjSubproblemX1)
    else:
        betaVarsCoeffObjSubproblem = copy.deepcopy(betaVarsCoeffObjSubproblemX0)

    # Parametric_columns:
    parametric_columns: list[tuple[list[str]]] = (
        InitScalable.defineParametricColumn(M=M, N=N)
    )
    return ColumnGenerationProblemBuilder(
        dataFrame=df,
        empiricalProbabilities=empiricalProbabilities,
        parametric_columns=parametric_columns,
        N=N,
        M=M,
        betaVarsCost=betaVarsCoeffObjSubproblem,
        betaVarsBitsX0=betaVarsBitsX0,
        betaVarsBitsX1=betaVarsBitsX1,
        interventionValue=interventionValue,
        minimum=minimum,
    )


def exemplo_de_execucao():
    N = 1
    M = 2
    scalable_df = get_scalable_dataframe(M=M, N=N)
    interventionValue = 1
    targetValue = 1

    scalarProblem = buildScalarProblem(
        M=M,
        N=N,
        interventionValue=interventionValue,
        targetValue=targetValue,
        df=scalable_df,
        minimum=True,
    )
    lower, itLower = scalarProblem.solve()

    scalarProblem = buildScalarProblem(
        M=M,
        N=N,
        interventionValue=interventionValue,
        targetValue=targetValue,
        df=scalable_df,
        minimum=False,
    )
    upper, itUpper = scalarProblem.solve()
    upper = -upper
    logger.info(f"{lower} =< P(Y = {targetValue}|X = {interventionValue}) <= {upper}")
    logger.info(f"{itLower} iteracoes para lower e {itUpper} para upper")


def main():
    return exemplo_de_execucao()