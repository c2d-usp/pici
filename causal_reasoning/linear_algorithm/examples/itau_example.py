import time as tm

from scipy.optimize import linprog
import pandas as pd

from causal_reasoning.linear_algorithm.mechanisms_generator import MechanismGenerator
from causal_reasoning.linear_algorithm.probabilities_helper import find_probability, find_conditional_probability
from causal_reasoning.graph.graph import Graph
from causal_reasoning.causal_model import get_graph, CausalModel
from causal_reasoning.utils._enum import Examples


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
        latentNode="U1", endogenousNodes=[
            "Y", "X"])
    
    y0: int = 1
    x0: int = 1
    xRlt: dict[str, int] = {}
    yRlt: dict[str, int] = {}
    dRlt: dict[str, int] = {}
    dxRlt: dict[str, int] = {}
    c: list[float] = []
    a: list[list[float]] = []
    b: list[float] = []
    df: pd.DataFrame = pd.read_csv(
        Examples.CSV_ITAU_EXAMPLE.value)

    bounds: list[tuple[float]] = [(0, 1) for _ in range(len(mechanism))]

    xRlt["X"] = x0

    for u in range(len(mechanism)):
        coef = 0
        for d in range(2):
            dRlt["D"] = d
            if mechanism[u]["X=" + str(x0) + ",D=" + str(d)] == y0:
                coef += find_conditional_probability(
                    dataFrame=df,
                    targetRealization=dRlt,
                    conditionRealization=xRlt)
        c.append(coef)
    a.append([1 for _ in range(len(mechanism))])
    b.append(1)
    for y in range(2):
        for x in range(2):
            for d in range(2):
                aux: list[float] = []
                yRlt["Y"] = y
                dxRlt["X"] = x
                dxRlt["D"] = d
                xRlt["X"] = x
                dRlt["D"] = d
                b.append(
                    find_conditional_probability(
                        dataFrame=df,
                        targetRealization=yRlt,
                        conditionRealization=dxRlt) *
                    find_probability(
                        dataFrame=df,
                        variableRealizations=xRlt))
                for u in range(len(mechanism)):
                    if (mechanism[u]["X=" + str(x) + ",D=" + str(d)]
                            == y) and (mechanism[u][""] == x):
                        aux.append(1)
                    else:
                        aux.append(0)
                a.append(aux)
    for i in range(len(a)):
        print(f"{a[i]} = {b[i]}")
    opt_problem(objFunction=c, Aeq=a, Beq=b, interval=bounds, v=True)


if __name__ == "__main__":
    itau_input = (
        "X -> Y, X -> D, D -> Y, E -> D, U1 -> Y, U1 -> X, U2 -> D, U3 -> E, U1 -> F"
    )
    itau_unobs = ["U1", "U2", "U3"]
    itau_target = "Y"
    itau_intervention = "X"
    itau_csv_path = Examples.CSV_ITAU_EXAMPLE.value
    itau_df = pd.read_csv(itau_csv_path)

    itau_model = CausalModel(
        data=itau_df,
        edges=itau_input,
        unobservables=itau_unobs,
        interventions=itau_intervention,
        interventions_value=1,
        target=itau_target,
        target_value=1,
    )
    dag = itau_model.graph


    # itau_input = (
    #     "X -> Y, X -> D, D -> Y, E -> D, U1 -> Y, U1 -> X, U2 -> D, U3 -> E, U1 -> F"
    # )
    # itau_unobs = ["U1", "U2", "U3"]
    # itau_target = "Y"
    # itau_intervention = "X"
    # dag = get_graph(str_graph=itau_input, unobservables=itau_unobs)  # use itau_simplified
    start = tm.time()
    main(dag=dag)
    end = tm.time()
    print(f"Time taken {end - start}")
