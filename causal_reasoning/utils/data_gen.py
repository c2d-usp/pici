import random
import pandas as pd


def f(a: int, b: int):
    return 1 if random.randrange(100) < 5 else a ^ b

def fx(u: int):
    return int(not u) if random.randrange(100) < 2 else u

def fa(ai: int, b: list[int], u: int):
    b_and = 1
    for bi in b:
        b_and = 1 if bi else 0
    return (b_and ^ u) ^ ai


def generate_data_for_scale_case(n: int, m: int, samples: int = 10000):
    file_path = f"./causal_reasoning/data/n{n}_m{m}_scaling_case.csv"
    U1 = [random.choice([0, 1]) for _ in range(samples)]
    U2 = [random.choice([0, 1]) for _ in range(samples)]
    X = [fx(u) for u in U1]

    B = []
    A = []

    B1 = [f(X[i], U2[i]) for i in range(samples)]
    A1 = [f(B1[i], U1[i]) for i in range(samples)]
    B.append(B1)
    A.append(A1)

    # For column output
    columns_values = {
        'U1': U1,
        'U2': U2,
        'X': X,
        'B1': B1,
        'A1': A1
    }
    for k in range(1, m):
        Bi = [f(X[j], U2[j]) for j in range(samples)]
        B.append(Bi)
        columns_values[f'B{k + 1}'] = Bi

    for i in range(1, n):
        Ai = [fa(A[i-1][j], [row[j] for row in B] , U1[j]) for j in range(samples)]
        A.append(Ai)
        columns_values[f'A{i + 1}'] = Ai

    last_A = A[-1]
    Y = [f(last_A[j], U2[j]) for j in range(samples)]
    columns_values['Y'] = Y

    df = pd.DataFrame(columns_values)
    df.to_csv(file_path, index=False)

if __name__ == "__main__":
    n = 6
    m = 6
    for i in range(1, n+1):
        for j in range(1, m+1):
            generate_data_for_scale_case(i, j)
    