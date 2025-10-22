import os
import random

import numpy as np
import pandas as pd

THIS_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
import sys

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def binary_f(a: int, b: int):
    return 1 if random.randrange(100) < 5 else a ^ b


def binary_fx(u: int):
    return int(not u) if random.randrange(100) < 2 else u


def binary_fa(ai: int, b: list[int], u: int):
    b_and = 1
    for bi in b:
        b_and = 1 if bi else 0
    return (b_and ^ u) ^ ai


def generate_data_for_scale_case(n: int, m: int, samples: int = 10000):
    file_path = f"./pici/data/csv/n{n}_m{m}_scaling_case.csv"
    U1 = [random.choice([0, 1]) for _ in range(samples)]
    U2 = [random.choice([0, 1]) for _ in range(samples)]
    X = [binary_fx(u) for u in U1]

    B = []
    A = []

    B1 = [binary_f(X[i], U2[i]) for i in range(samples)]
    A1 = [binary_f(B1[i], U1[i]) for i in range(samples)]
    B.append(B1)
    A.append(A1)

    # For column output
    columns_values = {"U1": U1, "U2": U2, "X": X, "B1": B1, "A1": A1}
    for k in range(1, m):
        Bi = [binary_f(X[j], U2[j]) for j in range(samples)]
        B.append(Bi)
        columns_values[f"B{k + 1}"] = Bi

    for i in range(1, n):
        Ai = [binary_fa(A[i - 1][j], [row[j] for row in B], U1[j]) for j in range(samples)]
        A.append(Ai)
        columns_values[f"A{i + 1}"] = Ai

    last_A = A[-1]
    Y = [binary_f(last_A[j], U2[j]) for j in range(samples)]
    columns_values["Y"] = Y

    df = pd.DataFrame(columns_values)
    df.to_csv(file_path, index=False)


def generate_digraph_data(card_z, card_x, card_y, n_samples=10000, seed=None):
    """
    Generate a dataset from the causal graph:
        z → x → y
        u → x, u → y
    where:
      - u is binary (0/1)
      - z, x, y are categorical with given cardinalities
      - relationships are deterministic (modular arithmetic)
    """
    file_path = f"{PROJECT_ROOT}/pici/data/csv/z{card_z}_x{card_x}_y{card_y}_discrete_iv.csv"
    if seed is not None:
        np.random.seed(seed)

    # Random distributions for exogenous nodes
    u = np.random.randint(0, 2, size=n_samples)
    z = np.random.randint(0, card_z, size=n_samples)

    x = (z + u) % (card_x)

    y = (x + u) % (card_y)

    # Combine into DataFrame
    df = pd.DataFrame({
        'Z': z,
        'X': x,
        'Y': y
    })

    # Save to CSV
    df.to_csv(file_path, index=False)


if __name__ == "__main__":
    # n = 6
    # m = 6
    # for i in range(1, n + 1):
    #     for j in range(1, m + 1):
    #         generate_data_for_scale_case(i, j)

    generate_digraph_data(card_z=2, card_x=2, card_y=2, seed=42)
    # generate_digraph_data(card_z=4, card_x=2, card_y=2)