from scipy.optimize import linprog


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

    return lowerBound, upperBound


def main():
    """
    Balke & Pearl IV example: code all the restrictions and use a linear solver.
    Variables are: Qij, with i and j in {0, 1, 2, 3}
    """

    # objective for: ACE(D - Y) = P(v=l) - P(ry=2) = sum(Qi1) - sum(Qi2)
    c = [0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 0]
    # P(ry=l) + P(ry=3) = sum(Qi1) + sum(Qi3)
    c1 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    # P(ry=2) + P(ry=3) = sum(Qi2) + sum(Qi3)
    c2 = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]

    bounds = [(0, 1) for _ in range(16)]

    a_eq = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1]
            ]

    b_eq = [1, 0.32, 0.32, 0.04, 0.32, 0.02, 0.17, 0.67, 0.14]

    # ---------- compute using the final objective function -------- #
    print("\nUsing the complete objective function, the results are:")
    lb0, ub0 = opt_problem(c, a_eq, b_eq, bounds, False)
    print(f"Lower bound: {lb0} - Upper bound: {ub0}")

    # ---------- compute using the partial objective functions -------- #
    print("\nUsing the complete objective function, the result for the positive query is:")
    lb1, ub1 = opt_problem(c1, a_eq, b_eq, bounds, False)
    print(f"Lower bound: {lb1} - Upper bound: {ub1}")

    print("Using the complete objective function, the result for the negative query is:")
    lb2, ub2 = opt_problem(c2, a_eq, b_eq, bounds, False)
    print(f"Lower bound: {lb2} - Upper bound: {ub2}")

    # ---------- results comparison -------- #
    print(f"\nWith the first method, we obtain the interval: [{lb0},{ub0}]")
    print(
        f"With the second method, we obtain the interval: [{trim_decimal(3,lb1 - ub2)},{trim_decimal(3,ub1 - lb2)}]")


if __name__ == "__main__":
    main()
