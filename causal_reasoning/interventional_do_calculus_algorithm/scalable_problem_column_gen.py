import copy
import gurobipy as gp
from gurobipy import GRB
import logging

logger = logging.getLogger(__name__)


import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
from itertools import product

from causal_reasoning.utils.probabilities_helper import (
    find_conditional_probability2,
    find_probability2,
)
from causal_reasoning.interventional_do_calculus_algorithm.scalable_problem_init import (
    InitScalable,
)
from causal_reasoning.utils.get_scalable_df import getScalableDataFrame

BIG_M = 1e4
DBG = False


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


class SubProblem:
    def __init__(self, N: int, M: int):
        self.model = gp.Model("subproblem")
        self.bit0 = {}  # X mechanism
        self.clusterBits = [
            {} for _ in range(N + 1)
        ]  # Bits for the mechanisms of A1,A2,..,An
        self.beta_varsX0 = {}
        self.beta_varsX1 = {}
        self.bitsParametric = [{} for _ in range((1 << (N + M + 1)))]
        self.constr = None

    def setup(
        self,
        amountBitsPerCluster: int,
        amountBetaVarsPerX: int,
        duals: list[float],
        amountNonTrivialRestrictions: int,
        betaVarsCost: list[float],
        parametric_column: list[tuple[list[int]]],
        betaVarsBitsX0: list[tuple[list[str]]],
        betaVarsBitsX1: list[tuple[list[str]]],
        N: int,
        M: int,
        interventionValue: int,
        minimum: bool,
    ):

        # Bit that determines the value of X.
        self.bit0 = self.model.addVars(1, obj=0, vtype=GRB.BINARY, name=["bit0"])

        # N bit clusters, one for each A_i (i = 1,2..,N)
        for clusterIndex in range(1, N + 1):
            clusterBitsNames: list[str] = []
            for i in range(amountBitsPerCluster):
                clusterBitsNames.append(f"cluster_{clusterIndex}_bit_{i}")

            self.clusterBits[clusterIndex] = self.model.addVars(
                amountBitsPerCluster, obj=0, vtype=GRB.BINARY, name=clusterBitsNames
            )

        # Beta Var when X=0:
        betaVarX0Names: list[str] = []
        for i in range(amountBetaVarsPerX):
            betaVarX0Names.append(f"BX0_{i}")
        if minimum:
            self.beta_varsX0 = self.model.addVars(
                amountBetaVarsPerX,
                obj=[cost * (1 - interventionValue) for cost in betaVarsCost],
                vtype=GRB.BINARY,
                name=betaVarX0Names,
            )
        else:
            self.beta_varsX0 = self.model.addVars(
                amountBetaVarsPerX,
                obj=[-cost * (1 - interventionValue) for cost in betaVarsCost],
                vtype=GRB.BINARY,
                name=betaVarX0Names,
            )

        # Beta Var when X=1:
        betaVarX1Names: list[str] = []
        for i in range(amountBetaVarsPerX):
            betaVarX1Names.append(f"BX1_{i}")

        if minimum:
            self.beta_varsX1 = self.model.addVars(
                amountBetaVarsPerX,
                obj=[cost * interventionValue for cost in betaVarsCost],
                vtype=GRB.BINARY,
                name=betaVarX1Names,
            )
        else:
            self.beta_varsX1 = self.model.addVars(
                amountBetaVarsPerX,
                obj=[-cost * interventionValue for cost in betaVarsCost],
                vtype=GRB.BINARY,
                name=betaVarX1Names,
            )
        # Parametric Columns variables:
        parametricColumnsNames: list[str] = []
        for i in range(amountNonTrivialRestrictions):
            parametricColumnsNames.append(f"Parametric{i}")

        self.bitsParametric = self.model.addVars(
            amountNonTrivialRestrictions,
            obj=[-duals[dualKey] for dualKey in duals],
            vtype=GRB.BINARY,
            name=parametricColumnsNames,
        )

        # Constraints for beta VarX0:
        for betaVarX0Index in range(len(betaVarsBitsX0)):
            self.model.addConstr(
                self.beta_varsX0[betaVarX0Index] >= 0, name=f"BetaX0_{betaVarX0Index}"
            )
            self.model.addConstr(
                self.beta_varsX0[betaVarX0Index] <= 1, name=f"BetaX0_{betaVarX0Index}"
            )

            for bitPlus in betaVarsBitsX0[betaVarX0Index][0]:
                parts = bitPlus.split("_")
                clusterBitIndex = int(parts[0][1:])
                bitIndex = int(parts[1])
                self.model.addConstr(
                    self.beta_varsX0[betaVarX0Index]
                    <= self.clusterBits[clusterBitIndex][bitIndex],
                    name=f"BetaX0_{betaVarX0Index}_BitPlus_{bitPlus}",
                )

            for bitMinus in betaVarsBitsX0[betaVarX0Index][1]:
                parts = bitMinus.split("_")
                clusterBitIndex = int(parts[0][1:])
                bitIndex = int(parts[1])
                self.model.addConstr(
                    self.beta_varsX0[betaVarX0Index]
                    <= 1 - self.clusterBits[clusterBitIndex][bitIndex],
                    name=f"BetaX0_{betaVarX0Index}_BitMinus_{bitMinus}",
                )

        self.constrs = self.model.addConstrs(
            (
                gp.quicksum(
                    self.clusterBits[int(bitPlus.split("_")[0][1:])][
                        int(bitPlus.split("_")[1])
                    ]
                    for bitPlus in betaVarsBitsX0[indexBetaVarX0][0]
                )
                + gp.quicksum(
                    1
                    - self.clusterBits[int(bitMinus.split("_")[0][1:])][
                        int(bitMinus.split("_")[1])
                    ]
                    for bitMinus in betaVarsBitsX0[indexBetaVarX0][1]
                )
                + 1
                - (
                    len(betaVarsBitsX0[indexBetaVarX0][0])
                    + len(betaVarsBitsX0[indexBetaVarX0][1])
                )
                <= self.beta_varsX0[indexBetaVarX0]
                for indexBetaVarX0 in range(len(self.beta_varsX0))
            ),
            name="betaX0_Force1",
        )

        # ------ Constraints for beta VarX1:
        for betaVarX1Index in range(len(betaVarsBitsX1)):
            self.model.addConstr(
                self.beta_varsX1[betaVarX1Index] >= 0, name=f"BetaX1_{betaVarX1Index}"
            )
            self.model.addConstr(
                self.beta_varsX1[betaVarX1Index] <= 1, name=f"BetaX1_{betaVarX1Index}"
            )

            for bitPlus in betaVarsBitsX1[betaVarX1Index][0]:
                parts = bitPlus.split("_")
                clusterBitIndex = int(parts[0][1:])
                bitIndex = int(parts[1])
                self.model.addConstr(
                    self.beta_varsX1[betaVarX1Index]
                    <= self.clusterBits[clusterBitIndex][bitIndex],
                    name=f"BetaX1_{betaVarX1Index}_BitPlus_{bitPlus}",
                )

            for bitMinus in betaVarsBitsX1[betaVarX1Index][1]:
                parts = bitMinus.split("_")
                clusterBitIndex = int(parts[0][1:])
                bitIndex = int(parts[1])
                self.model.addConstr(
                    self.beta_varsX1[betaVarX1Index]
                    <= 1 - self.clusterBits[clusterBitIndex][bitIndex],
                    name=f"BetaX1_{betaVarX1Index}_BitMinus_{bitMinus}",
                )

        self.constrs = self.model.addConstrs(
            (
                gp.quicksum(
                    self.clusterBits[int(bitPlus.split("_")[0][1:])][
                        int(bitPlus.split("_")[1])
                    ]
                    for bitPlus in betaVarsBitsX1[indexBetaVarX1][0]
                )
                + gp.quicksum(
                    1
                    - self.clusterBits[int(bitMinus.split("_")[0][1:])][
                        int(bitMinus.split("_")[1])
                    ]
                    for bitMinus in betaVarsBitsX1[indexBetaVarX1][1]
                )
                + 1
                - (
                    len(betaVarsBitsX1[indexBetaVarX1][0])
                    + len(betaVarsBitsX1[indexBetaVarX1][1])
                )
                <= self.beta_varsX1[indexBetaVarX1]
                for indexBetaVarX1 in range(len(self.beta_varsX1))
            ),
            name="betaX1_Force1",
        )

        # ------ Constraints for parametric columns in function of beta vars and bit0 (X)
        for indexParametric in range(amountNonTrivialRestrictions):
            self.model.addConstr(
                self.bitsParametric[indexParametric] >= 0,
                name=f"ParametricPositive{indexParametric}",
            )
            self.model.addConstr(
                self.bitsParametric[indexParametric] <= 1,
                name=f"ParametricUpper{indexParametric}",
            )

            # beta0_{restrictionIndex} b0  beta1_{restrictionIndex - (1 << M + N)}
            for bitPlus in parametric_column[indexParametric][0]:
                if len(bitPlus) == 2:  # b0
                    self.model.addConstr(
                        self.bitsParametric[indexParametric] <= self.bit0[0],
                        name=f"Parametric_{indexParametric}BitPlus{bitPlus}",
                    )
                elif bitPlus[4] == "0":  # beta 0
                    self.model.addConstr(
                        self.bitsParametric[indexParametric]
                        <= self.beta_varsX0[int(bitPlus.split("_")[-1])],
                        name=f"Parametric_{indexParametric}BitPlus{bitPlus}",
                    )
                else:  # beta 1
                    self.model.addConstr(
                        self.bitsParametric[indexParametric]
                        <= self.beta_varsX1[int(bitPlus.split("_")[-1])],
                        name=f"Parametric_{indexParametric}BitPlus{bitPlus}",
                    )

            for bitMinus in parametric_column[indexParametric][1]:
                if len(bitMinus) == 2:  # b0
                    self.model.addConstr(
                        self.bitsParametric[indexParametric] <= 1 - self.bit0[0],
                        name=f"Parametric_{indexParametric}bitMinus{bitMinus}",
                    )
                elif bitMinus[4] == "0":  # beta 0
                    self.model.addConstr(
                        self.bitsParametric[indexParametric]
                        <= 1 - self.beta_varsX0[int(bitMinus.split("_")[-1])],
                        name=f"Parametric_{indexParametric}bitMinus{bitMinus}",
                    )
                else:  # beta 1
                    self.model.addConstr(
                        self.bitsParametric[indexParametric]
                        <= 1 - self.beta_varsX1[int(bitMinus.split("_")[-1])],
                        name=f"Parametric_{indexParametric}bitMinus{bitMinus}",
                    )

        # 1 - N + sum(b+) + sum(1 - b-) <= beta
        self.constrs = self.model.addConstrs(
            (
                gp.quicksum(
                    self.bit0[0] * (len(bitPlus) == 2)
                    + self.beta_varsX0[
                        (
                            int((bitPlus + "00000")[6:]) // 100_000
                            if len(bitPlus) > 2
                            else 0
                        )
                    ]
                    * ((bitPlus + "22222")[4] == "0")
                    + self.beta_varsX1[
                        (
                            int((bitPlus + "00000")[6:]) // 100_000
                            if len(bitPlus) > 2
                            else 0
                        )
                    ]
                    * ((bitPlus + "22222")[4] == "1")
                    for bitPlus in parametric_column[indexParametric][0]
                )
                + gp.quicksum(
                    1
                    - self.bit0[0] * (len(bitMinus) == 2)
                    - self.beta_varsX0[
                        (
                            int((bitMinus + "00000")[6:]) // 100_000
                            if len(bitMinus) > 2
                            else 0
                        )
                    ]
                    * ((bitMinus + "22222")[4] == "0")
                    - self.beta_varsX1[
                        (
                            int((bitMinus + "00000")[6:]) // 100_000
                            if len(bitMinus) > 2
                            else 0
                        )
                    ]
                    * ((bitMinus + "22222")[4] == "1")
                    for bitMinus in parametric_column[indexParametric][1]
                )
                + 1
                - (
                    len(parametric_column[indexParametric][0])
                    + len(parametric_column[indexParametric][1])
                )
                <= self.bitsParametric[indexParametric]
                for indexParametric in range(len(self.bitsParametric))
            ),
            name="ParametricForce1Condition",
        )

        # ----- END INTEGER PROGRAMMING CONSTRAINTS -----

        self.model.modelSense = GRB.MINIMIZE
        # Turning off output because of the iterative procedure
        # self.model.setParam('FeasibilityTol', 1e-9)
        self.model.params.outputFlag = 0
        self.model.params.Method = 4
        # Stop the subproblem routine as soon as the objective's best bound becomes
        # less than or equal to one, as this implies a non-negative reduced cost for
        # the entering column.
        self.model.params.bestBdStop = 1
        self.model.update()

    def update(self, duals):
        """
        Change the objective functions coefficients.
        """
        self.model.setAttr(
            "obj", self.bitsParametric, [-duals[dualKey] for dualKey in duals]
        )
        self.model.update()


class ScalarProblem:
    def __init__(
        self,
        dataFrame,
        empiricalProbabilities: list[float],
        parametric_columns: dict[str, tuple[list[int]]],
        N: int,
        M: int,
        betaVarsCost: list[float],
        betaVarsBitsX0: list[tuple[str]],
        betaVarsBitsX1: list[tuple[str]],
        interventionValue: int,
        minimum: bool,
    ):

        self.M = M
        self.N = N
        self.amountNonTrivialRestrictions = 1 << (M + N + 1)
        self.amountBitsPerCluster = 1 << (M + 1)
        self.amountBetaVarsPerX = 1 << (M + N)
        self.columns_base = None

        # Order parametric_columns (XA1A2..AnB1...Bm)
        self.empiricalProbabilities: list[float] = empiricalProbabilities
        self.parametric_columns: dict[str, tuple[list[int]]] = parametric_columns
        self.dataFrame = dataFrame
        self.betaVarsBitsX0 = betaVarsBitsX0
        self.betaVarsBitsX1 = betaVarsBitsX1
        self.betaVarsCost = betaVarsCost
        self.interventionValue = interventionValue  # X = x in {0, 1}
        self.minimum = minimum
        auxDict = {}
        for i in range(self.amountNonTrivialRestrictions):
            auxDict[i] = BIG_M
        self.duals = auxDict.copy()

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
            interventionValue=self.interventionValue,
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
                if self.interventionValue == 0:
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
            if counter >= 10000:
                raise TimeoutError(f"Too many iterations")
            logger.info(f"Iteration Number = {counter}")

        return counter

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
        return ScalarProblem(
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

    def solve(self):
        """
        Gurobi does not support branch-and-price, as this requires to add columns
        at local nodes of the search tree. A heuristic is used instead, where the
        integrality constraints for the variables of the final root LP relaxation
        are installed and the resulting MIP is solved. Note that the optimal
        solution could be overlooked, as additional columns are not generated at
        the local nodes of the search tree.
        """
        numberIterations = self._generate_patterns()
        self.master.model.setAttr("vType", self.master.vars, GRB.CONTINUOUS)  # useless?
        self.master.model.optimize()
        self.master.model.write("model.lp")
        self.master.model.write("model.mps")
        bound = self.master.model.ObjVal
        itBound = numberIterations
        return bound, itBound


def single_exec():
    N = 1
    M = 2
    scalable_df = getScalableDataFrame(M=M, N=N)
    interventionValue = 1
    targetValue = 1

    scalarProblem = ScalarProblem.buildScalarProblem(
        M=M,
        N=N,
        interventionValue=interventionValue,
        targetValue=targetValue,
        df=scalable_df,
        minimum=True,
    )
    lower, itLower = scalarProblem.solve()

    scalarProblem = ScalarProblem.buildScalarProblem(
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
    return single_exec()


if __name__ == "__main__":
    main()
