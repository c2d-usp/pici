import os
import sys

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import array

import gurobipy as gp
from gurobipy import GRB
import numpy as np


class PhaseI:
    def __init__(self):
        pass

    def phase_one_gurobi(self, A, b) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform Phase I of the Two-Phase Simplex method.
        A: m x n constraint matrix
        b: m vector (RHS), assumed to be >= 0 or will be made so
        """

        A = _parse_matrix_to_np_ndarray(A)
        b = _parse_matrix_to_np_ndarray(b)

        m, n = A.shape

        # Step: 1 Ensure b >= 0 by flipping rows if needed
        A_mod = A.copy()
        b_mod = b.copy()
        for i in range(m):
            if b_mod[i] < 0:
                A_mod[i, :] *= -1
                b_mod[i] *= -1

        # Create Gurobi model for Phase I
        model = gp.Model("phase_one")
        model.setParam("OutputFlag", 1)

        # Original variables
        x = model.addVars(n, lb=0.0, name="x")

        # Step: 2 Artificial variables
        artificial = model.addVars(m, lb=0.0, name="artificial")

        # Constraints: A_mod @ x + artificial = b_mod
        for i in range(m):
            constr_expr = (
                gp.quicksum(A_mod[i, j] * x[j] for j in range(n)) + artificial[i]
            )
            model.addConstr(constr_expr == b_mod[i], name=f"c_{i}")

        # Objective: minimize sum of artificial variables
        model.setObjective(gp.quicksum(artificial[i] for i in range(m)), GRB.MINIMIZE)

        # Solve using primal or dual simplex
        model.setParam("Method", 1)

        # Optimize Phase I
        model.optimize()

        if model.Status != GRB.OPTIMAL:
            print(f"Optimization failed or stopped with status: {model.Status}")
            return None, None, None

        # Step 3: Check if auxiliary costs are positive
        if model.ObjVal > 1e-6:
            print("Original problem is infeasible (artificial objective > 0).")
            return None, None, None

        # Step 4: Check which artificial variables are in the basis with value 0
        redundant_rows = []
        for i in range(m):
            if artificial[i].VBasis == 0:
                print(
                    f"Artificial var artificial[{i}] is in basis with value 0 — constraint {i} is redundant."
                )
                redundant_rows.append(i)

        # Remove redundant constraints from A and b
        A_clean = np.delete(A_mod, redundant_rows, axis=0)
        b_clean = np.delete(b_mod, redundant_rows, axis=0)

        # Extract feasible x solution (feasible solution to original Ax = b)
        x_vals = np.array([x[j].X for j in range(n)])

        return x_vals, A_clean, b_clean


def _parse_matrix_to_np_ndarray(M):
    if isinstance(M, np.ndarray):
        return M
    if isinstance(M, list):
        return np.array(M)
    if isinstance(M, array.array):
        return np.array(M)
    raise Exception("Unknown type.")


def main():

    # Se tiver variável artificial com coef zero na base, é simplesmente tirar? Parece que sim.
    # ---> Quando pode ter uma artficial com coef zero na base?
    # ----------> Se tem restrições LD, então haverá artificial com zero coef
    # ----------> Se tem var artificial com coef zero, então tem restrições LD?? (NÃO SEI)
    A = [[1, 2, 3, 0], [-1, 2, 6, 0], [0, 4, 9, 0], [0, 0, 3, 1]]

    b = [3, 2, 5, 1]
    x, A, b = PhaseI().phase_one_gurobi(A, b)
    print("Variables coefficients:")
    print(x)

    A = np.array([[1, 1], [2, 2]], dtype=float)

    b = np.array([4, 8], dtype=float)
    x, A, b = PhaseI().phase_one_gurobi(A, b)
    print("Variables coefficients:")
    print(x)

    A = np.array([[1, 1], [1, 1]], dtype=float)

    b = np.array([1, 3], dtype=float)

    x, A, b = PhaseI().phase_one_gurobi(A, b)
    print("Variables coefficients:")
    print(x)


if __name__ == "__main__":
    main()


# class TestParser(unittest.TestCase):
#     def tests(self):
#         A = [
#             [1, 2, 3, 0],
#             [-1, 2, 6, 0],
#             [0, 4, 9, 0],
#             [0, 0, 3, 1]
#         ]

#         b = [3,2,5,1]
#         x, A, b = PhaseI().phase_one_gurobi(A, b)
#         expected = [[ 1,  2,  3,  0],
#                     [-1,  2,  6,  0],
#                     [ 0,  4,  9,  0],
#                     [ 0,  0,  3,  1]]
#         self.assertListEqual(x, expected)

#         A = np.array([
#             [1, 1],
#             [2, 2]
#         ], dtype=float)

#         b = np.array([4, 8], dtype=float)
#         x, A, b = PhaseI().phase_one_gurobi(A, b)
#         expected = [[2., 2.]]
#         self.assertEqual(x, expected)

#         A = np.array([
#             [1, 1],
#             [1, 1]
#         ], dtype=float)

#         b = np.array([1, 3], dtype=float)
#         x, A, b = PhaseI().phase_one_gurobi(A, b)
#         expected = None
#         self.assertEqual(x, expected)


# if __name__ == '__main__':
#     unittest.main()
