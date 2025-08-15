import logging
import os
import sys

import pandas as pd

from pici.causal_model import CausalModel
from pici.utils._enum import DataExamplesPaths

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO)

from pici.graph.graph import Graph
from pici.graph.node import Node
from pici.intervention_inference_algorithm.column_generation.generic import bits
from pici.intervention_inference_algorithm.linear_programming.obj_function_generator import ObjFunctionGenerator


def calculate_optimization_problem(
    graph: Graph,
    df: pd.DataFrame,
    intervention: Node,
    target: Node,
) -> tuple[str, str]:
    objective_function = ObjFunctionGenerator(
        graph=graph,
        dataFrame=df,
        intervention=intervention,
        target=target,
    )

    symbolic_objective_function_probabilites: list[tuple] = objective_function.generate_symbolic_objective_function_probabilities()
    print(f"symbolic_objective_function_probabilites: {symbolic_objective_function_probabilites}")

    symbolic_decision_function: dict[tuple, int] = objective_function.generate_symbolic_decision_function()
    print(f"symbolic_decision_function: {symbolic_decision_function}")

    bits_list = bits.generate_optimization_problem_bit_list(intervention)
    print(f"bits_list: {bits_list}")
    # step 2: initialization()
    # step 3: lower, upper = solve()

def main():
    edges = "Z -> X, X -> Y, U1 -> X, U1 -> Y, U2 -> Z"
    card = {"Z": 2, "X": 2, "Y": 2, "U1": 0, "U2": 0}
    unobs = ["U1", "U2"]
    df = pd.read_csv(
        os.path.join(PROJECT_ROOT, DataExamplesPaths.CSV_BALKE_PEARL_EXAMPLE.value)
    )

    model = CausalModel(
        data=df, edges=edges, custom_cardinalities=card, unobservables_labels=unobs
    )
    model.set_interventions([("X", 1)])
    model.set_target(("Y", 1))
    calculate_optimization_problem(graph=model.graph, df=df, intervention=model.interventions[0], target=model.target)

if __name__ == "__main__":
    main()