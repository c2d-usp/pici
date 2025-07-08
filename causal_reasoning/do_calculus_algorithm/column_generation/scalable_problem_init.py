import os
import sys
from itertools import product

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from causal_reasoning.utils._enum import DataExamplesPaths
from causal_reasoning.utils.probabilities_helper import (
    find_conditional_probability2,
    find_probability2,
)


class InitScalable:
    def calculateEmpiricals(M: int, N: int, df, DBG=False):
        empiricalProbabilities: list[float] = []
        setCurrentVars: set[str] = {"X"}
        for i in range(1, N + 1):
            setCurrentVars.update({f"A{i}", f"B{i}"})

        for realizationCase in list(product([0, 1], repeat=N + M + 1)):
            currentProbability: float = find_probability2(
                dataFrame=df, realizationDict={"X": realizationCase[0]}
            )

            targetRealization: dict[str, int] = {}
            conditionRealization: dict[str, int] = {}
            for Aindex in range(1, N + 1):
                targetRealization[f"A{Aindex}"] = realizationCase[Aindex]

            conditionRealization["X"] = realizationCase[0]
            for ShiftedBindex in range(N + 1, N + M + 1):
                conditionRealization[f"B{ShiftedBindex - N}"] = realizationCase[
                    ShiftedBindex
                ]

            currentProbability *= find_conditional_probability2(
                dataFrame=df,
                targetRealization=targetRealization,
                conditionRealization=conditionRealization,
            )

            empiricalProbabilities.append(currentProbability)
            # if DBG: #print(f"Calculated empirical prob: {empiricalProbabilities[-1]}")

        empiricalProbabilities.append(1)
        return empiricalProbabilities

    def defineGammaUAuxiliaryVariables(
        M: int, N: int, df, XValue: int, targetValue: int, DBG=False
    ):
        betaVarsBits: list[tuple[list[str]]] = [0] * (
            1 << (N + M)
        )  # For each: (B+, B-), with B+ = list[str]
        betaVarsCoeffObjSubproblem: list[float] = [0] * (1 << (N + M))

        bitPlus: list[int] = []
        bitMinus: list[int] = []

        # Order of realization: (A1,A2,..,An,B1,B2,...,Bm)
        for betaVarIndex, realizationCase in enumerate(
            list(product([0, 1], repeat=M + N))
        ):
            # Pre-compute B contribution:
            bitIndexLower = 0
            indexBs = N + M - 1
            helper = 1
            while indexBs >= N:
                bitIndexLower += helper * realizationCase[indexBs]
                helper *= 2
                indexBs -= 1

            for clusterBitIndex in range(1, N + 1):
                if clusterBitIndex == 1:  # A_0 = X
                    bitIndexHigher = (1 << M) * XValue
                else:
                    bitIndexHigher = (1 << M) * realizationCase[clusterBitIndex - 2]

                bitIndex = bitIndexHigher + bitIndexLower

                if realizationCase[clusterBitIndex - 1] == 1:
                    bitPlus.append(
                        f"A{clusterBitIndex}_{bitIndex}"
                    )  # First the mechanism and then the index in the mechanism bit cluster
                else:
                    bitMinus.append(f"A{clusterBitIndex}_{bitIndex}")

            # ----- Calculate the obj coeff:
            objCoeff: float = 1.0
            objCoeff *= find_conditional_probability2(
                dataFrame=df,
                targetRealization={"Y": targetValue},
                conditionRealization={f"A{N}": realizationCase[N - 1]},
            )

            targetRealization: dict[str, int] = {}
            for index in range(N, N + M):
                targetRealization[f"B{index - N + 1}"] = realizationCase[index]

            objCoeff *= find_conditional_probability2(
                dataFrame=df,
                targetRealization=targetRealization,
                conditionRealization={"X": XValue},
            )

            betaVarsBits[betaVarIndex] = (bitPlus.copy(), bitMinus.copy())
            betaVarsCoeffObjSubproblem[betaVarIndex] = objCoeff
            bitPlus.clear()
            bitMinus.clear()

        return betaVarsBits, betaVarsCoeffObjSubproblem

    # Realization Order: XA1A2..AnB1..Bm, com Bm o LSB.
    def defineParametricColumn(M: int, N: int):
        parametric_columns: list[tuple[list[str]]] = []
        bitPlus: list[int] = []
        bitMinus: list[int] = []
        for restrictionIndex in range((1 << N + M + 1)):
            if restrictionIndex < (1 << M + N):
                xRealization = 0
            else:
                xRealization = 1

            if xRealization == 0:
                bitPlus.append(f"beta0_{restrictionIndex}")
                bitMinus.append("b0")
            else:
                bitPlus.append(f"beta1_{restrictionIndex - (1 << M + N)}")
                bitPlus.append("b0")

            parametric_columns.append((bitPlus.copy(), bitMinus.copy()))
            bitPlus.clear()
            bitMinus.clear()

        return parametric_columns


def testEmpirical():
    copilot_csv_path = DataExamplesPaths.CSV_2SCALING.value
    df = pd.read_csv(copilot_csv_path)
    # print("--- Test empiricials ---")
    InitScalable.calculateEmpiricals(M=1, N=2, df=df, DBG=True)


def testBetaVars():
    copilot_csv_path = DataExamplesPaths.CSV_2SCALING.value
    df = pd.read_csv(copilot_csv_path)
    # print("\n\n --- Test Gamma variables --- ")
    betaVarsBits, betaVarsCoeffObjSubproblem = (
        InitScalable.defineGammaUAuxiliaryVariables(
            M=2, N=2, df=df, interventionValue=1, targetValue=1, DBG=False
        )
    )

    # for ind, betaVar in enumerate(betaVarsBits):
    #     print(f"For betaVar {ind}")
    #     print("Bit plus:")
    #     for bitPlus in betaVar[0]:
    #       print(f"{bitPlus} ", end="")
    #     print("\n")

    # #print("Bit minus:")
    # for bitMinus in betaVar[1]:
    #   print(f"{bitMinus} ", end="")
    # print("\n")

    # print("Check coefficients:")
    checkCoefs: list[float] = []
    for realizationCase in list(product([0, 1], repeat=4)):
        prob = find_conditional_probability2(
            dataFrame=df,
            conditionRealization={"A2": realizationCase[1]},
            targetRealization={"Y": 1},
        )
        prob *= find_conditional_probability2(
            dataFrame=df,
            conditionRealization={"X": 1},
            targetRealization={"B1": realizationCase[2], "B2": realizationCase[3]},
        )
        checkCoefs.append(prob)

    # for ind, coef in enumerate(betaVarsCoeffObjSubproblem):
    #   print(f"#{ind} coef = {coef} | check = {checkCoefs[ind]}")


def testParametricColumns():
    parametricColumns = InitScalable.defineParametricColumn(M=2, N=2)
    # for index, row in enumerate(parametricColumns):
    #   print(f"Analyse row {index}")
    #   print(row)


if __name__ == "__main__":
    # testEmpirical()
    # testBetaVars()
    testParametricColumns()
