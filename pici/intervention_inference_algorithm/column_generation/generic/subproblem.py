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
        duals: dict[int, float],
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

