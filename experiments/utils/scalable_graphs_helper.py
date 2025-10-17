from itertools import product

import pandas as pd

from pici.utils._enum import DataExamplesPaths
from pici.utils.probabilities_helper import (
    find_conditional_probability2,
    find_probability2,
)


def get_three_latents_scalable_dataframe(M: int, N: int):
    if N == 1 and M == 1:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N1M1.value
    elif N == 2 and M == 1:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N2M1.value
    elif N == 3 and M == 1:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N3M1.value
    elif N == 4 and M == 1:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N4M1.value
    elif N == 5 and M == 1:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N5M1.value
    elif N == 6 and M == 1:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N6M1.value
    elif N == 1 and M == 2:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N1M2.value
    elif N == 2 and M == 2:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N2M2.value
    elif N == 3 and M == 2:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N3M2.value
    elif N == 4 and M == 2:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N4M2.value
    elif N == 5 and M == 2:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N5M2.value
    elif N == 1 and M == 3:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N1M3.value
    elif N == 2 and M == 3:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N2M3.value
    elif N == 3 and M == 3:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N3M3.value
    elif N == 4 and M == 3:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N4M3.value
    elif N == 1 and M == 4:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N1M4.value
    elif N == 2 and M == 4:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N2M4.value
    elif N == 3 and M == 4:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N3M4.value
    elif N == 1 and M == 5:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N1M5.value
    elif N == 2 and M == 5:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N2M5.value
    elif N == 1 and M == 6:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N1M6.value

    return pd.read_csv(scalable_csv_path)


def generate_three_latents_scalable_string_edges(N, M) -> str:
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

def generate_three_latents_binary_scalable_cardinalities(N, M) -> dict:
    cardinalities = {"U1":0, "U2":0, "U3":0}
    for i in range(1, N + 1):
        cardinalities[f"A{i}"] = 2

    for i in range(1, M + 1):
        cardinalities[f"B{i}"] = 2
    return cardinalities

def find_true_value_in_three_latents_scalable_graphs(N: int, M: int, y0: int, x0: int, df):
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

def generate_two_latents_scalable_string_edges(N, M) -> str:
    scalable_input: str = "U1 -> X, U2 -> Y, "
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

def generate_two_latents_binary_scalable_cardinalities(N, M) -> dict:
    cardinalities = {"U1":0, "U2":0}
    for i in range(1, N + 1):
        cardinalities[f"A{i}"] = 2

    for i in range(1, M + 1):
        cardinalities[f"B{i}"] = 2
    return cardinalities

def get_two_latents_scalable_dataframe(M: int, N: int):
    if N == 1 and M == 1:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N1M1.value
    elif N == 2 and M == 1:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N2M1.value
    elif N == 3 and M == 1:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N3M1.value
    elif N == 4 and M == 1:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N4M1.value
    elif N == 5 and M == 1:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N5M1.value
    elif N == 6 and M == 1:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N6M1.value
    elif N == 1 and M == 2:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N1M2.value
    elif N == 2 and M == 2:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N2M2.value
    elif N == 3 and M == 2:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N3M2.value
    elif N == 4 and M == 2:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N4M2.value
    elif N == 5 and M == 2:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N5M2.value
    elif N == 1 and M == 3:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N1M3.value
    elif N == 2 and M == 3:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N2M3.value
    elif N == 3 and M == 3:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N3M3.value
    elif N == 4 and M == 3:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N4M3.value
    elif N == 1 and M == 4:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N1M4.value
    elif N == 2 and M == 4:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N2M4.value
    elif N == 3 and M == 4:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N3M4.value
    elif N == 1 and M == 5:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N1M5.value
    elif N == 2 and M == 5:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N2M5.value
    elif N == 1 and M == 6:
        scalable_csv_path = DataExamplesPaths.CSV_3_LATENTS_N1M6.value

    # return pd.read_csv(scalable_csv_path)
    pass

def get_example_input(N, M, case):
    if case == "three_latent":
        edges = generate_two_latents_scalable_string_edges(N, M)
        df = get_three_latents_scalable_dataframe(M=M, N=N)
        unobs = ["U1", "U2", "U3"]
    else:
        edges = generate_two_latents_scalable_string_edges(N, M)
        df = get_two_latents_scalable_dataframe(M=M, N=N)
        unobs = ["U1", "U2"]
    return edges, df, unobs