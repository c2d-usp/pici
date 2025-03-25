import os
import time as tm

from scipy.optimize import linprog
import pandas as pd

from causal_usp_icti.utils.mechanisms_generator import MechanismGenerator
from causal_usp_icti.utils.probabilities_helper import ProbabilitiesHelper
from causal_usp_icti.graph.graph import Graph
from causal_usp_icti.causal_model import get_graph
from causal_usp_icti.utils._enum import Examples


def trim_decimal(precision: int, value: float):
    return round(pow(10, precision) * value) / pow(10, precision)


def opt_problem(objFunction: list[float],
               Aeq: list[list[float]],
               Beq: list[float],
               interval,
               v: bool):
    lowerBoundSol = linprog(
        c=objFunction,
        A_ub=None,
        b_ub=None,
        A_eq=Aeq,
        b_eq=Beq,
        method="highs",
        bounds=interval)
    upperBoundSol = linprog(c=[-x for x in objFunction],
                            A_ub=None,
                            b_ub=None,
                            A_eq=Aeq,
                            b_eq=Beq,
                            method="highs",
                            bounds=interval)

    if lowerBoundSol.success:
        lowerBound = trim_decimal(3, lowerBoundSol.fun)
        if v:
            print(f"Optimal distribution = {lowerBoundSol.x}")
            print(f"Obj. function = {lowerBound}")
    else:
        print("Solution not found:", lowerBoundSol.message)

    # Find maximum (uses the negated objective function and changes the sign
    # of the result)
    if upperBoundSol.success:
        upperBound = trim_decimal(3, -upperBoundSol.fun)
        if v:
            print(f"Optimal distribution = {upperBoundSol.x}")
            print(f"Obj. function = {upperBound}")
    else:
        print("Solution not found:", upperBoundSol.message)
    print(f"[{lowerBound}, {upperBound}]")
    return lowerBound, upperBound


def main(dag: Graph):

    _, _, mechanism = MechanismGenerator.mechanisms_generator(
        latentNode=dag.labelToIndex["U1"], endogenousNodes=[
            dag.labelToIndex["Y"], dag.labelToIndex["X"]], cardinalities=dag.cardinalities, graphNodes=dag.graphNodes, v=False)
    y0: int = 1
    x0: int = 1
    xRlt: dict[int, int] = {}
    yRlt: dict[int, int] = {}
    dRlt: dict[int, int] = {}
    dxRlt: dict[int, int] = {}
    c: list[float] = []
    a: list[list[float]] = []
    b: list[float] = []
    df: pd.DataFrame = pd.read_csv(
        Examples.CSV_ITAU_EXAMPLE.value)

    bounds: list[tuple[float]] = [(0, 1) for _ in range(len(mechanism))]

    xRlt[dag.labelToIndex["X"]] = x0

    for u in range(len(mechanism)):
        coef = 0
        for d in range(2):
            dRlt[dag.labelToIndex["D"]] = d
            if mechanism[u]["3=" + str(x0) + ",4=" + str(d)] == y0:
                coef += ProbabilitiesHelper.find_conditional_probability(
                    dataFrame=df,
                    indexToLabel=dag.indexToLabel,
                    targetRealization=dRlt,
                    conditionRealization=xRlt)
        c.append(coef)
    a.append([1 for _ in range(len(mechanism))])
    b.append(1)
    for y in range(2):
        for x in range(2):
            for d in range(2):
                aux: list[float] = []
                yRlt[dag.labelToIndex["Y"]] = y
                dxRlt[dag.labelToIndex["X"]] = x
                dxRlt[dag.labelToIndex["D"]] = d
                xRlt[dag.labelToIndex["X"]] = x
                dRlt[dag.labelToIndex["D"]] = d
                b.append(
                    ProbabilitiesHelper.find_conditional_probability(
                        dataFrame=df,
                        indexToLabel=dag.indexToLabel,
                        targetRealization=yRlt,
                        conditionRealization=dxRlt) *
                    ProbabilitiesHelper.find_probability(
                        dataFrame=df,
                        indexToLabel=dag.indexToLabel,
                        variableRealizations=xRlt))
                for u in range(len(mechanism)):
                    if (mechanism[u]["3=" + str(x) + ",4=" + str(d)]
                            == y) and (mechanism[u][""] == x):
                        aux.append(1)
                    else:
                        aux.append(0)
                a.append(aux)
    for i in range(len(a)):
        print(f"{a[i]} = {b[i]}")
    opt_problem(objFunction=c, Aeq=a, Beq=b, interval=bounds, v=True)


if __name__ == "__main__":
    dag = get_graph(file=Examples.TXT_ITAU_EXAMPLE.value)  # use itau_simplified
    start = tm.time()
    main(dag=dag)
    end = tm.time()
    print(f"Time taken {end - start}")
