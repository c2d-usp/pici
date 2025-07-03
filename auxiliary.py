import time as tm
import logging

logger = logging.getLogger(__name__)

import pandas as pd
from itertools import product
from causal_reasoning.utils.probabilities_helper import (
    find_conditional_probability2,
    find_probability2,
)
from causal_reasoning.utils.get_scalable_df import getScalableDataFrame


def genGraph(N, M):
    scalable_input: str = "U1 -> X, U3 -> Y, "
    for i in range(1, N + 1):
        scalable_input += f"U1 -> A{i}, "
        if i == 1:
            scalable_input += "X -> A1, "
        else:
            scalable_input += f"A{i-1} -> A{i}, "
    scalable_input += f"A{N} -> Y, "

    for i in range(1, M + 1):
        scalable_input += f"U2 -> B{i}, "
        scalable_input += f"X -> B{i}, "
        for j in range(1, N + 1):
            scalable_input += f"B{i} -> A{j}, "

    return scalable_input[:-2]


def true_value(N, M, y0, x0, df):
    prob = 0
    for rlt in list(product([0, 1], repeat=2)):
        term = 1
        term *= find_conditional_probability2(
            dataFrame=df,
            targetRealization={"Y": y0},
            conditionRealization={f"A{N}": rlt[0]},
        )
        term *= find_conditional_probability2(
            dataFrame=df,
            targetRealization={f"A{N}": rlt[0]},
            conditionRealization={"U1": rlt[1], "X": x0},
        )
        term *= find_probability2(dataFrame=df, realizationDict={"U1": rlt[1]})
        prob += term
    return prob


def main():
    FILE_PATH = "./true_values.txt"
    N_M = [
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 4),
        (2, 5),  # (2,6),
        (3, 1),
        (3, 2),
        (3, 3),
        (3, 4),  # (3,5),(3,6),
        (4, 1),
        (4, 2),
        (4, 3),  # (4,4),(4,5),(4,6),
        (5, 1),
        (5, 2),  # (5,3),(5,4),(5,5),(5,6),
        (6, 1),  # (6,2),(6,3),(6,4),(6,5),(6,6),
    ]
    tuple_target_inter = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for value in N_M:
        N, M = value
        scalable_df = getScalableDataFrame(M=M, N=N)
        with open(FILE_PATH, "a") as file:
            file.write(f'n{N}_m{M}_scaling_case ==> Graph: ==> "{genGraph(N=N,M=M)}"\n')
        for target_value, intervention_value in tuple_target_inter:
            with open(FILE_PATH, "a") as file:
                t = true_value(N, M, target_value, intervention_value, scalable_df)
                file.write(
                    f"n{N}_m{M}_scaling_case - P(Y={target_value}|do(X={intervention_value})): {t}\n"
                )


if __name__ == "__main__":
    main()
