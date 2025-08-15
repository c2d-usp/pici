from pandas import DataFrame
from pici.graph.graph import Graph
from pici.graph.node import Node
from pici.intervention_inference_algorithm.column_generation.generic import bits
from pici.intervention_inference_algorithm.linear_programming.obj_function_generator import ObjFunctionGenerator


def calculate_optimization_problem(
    graph: Graph,
    df: DataFrame,
    intervention: Node,
    target: Node,
) -> tuple[str, str]:
    objective_function = ObjFunctionGenerator(
        graph=graph,
        dataFrame=df,
        intervention=intervention,
        target=target,
    )
    symbolic_objective_function: list[tuple] = objective_function.generate_symbolic_objective_function()
    symbolic_decision_function: dict[tuple, int] = objective_function.generate_symbolic_decision_function()
    bits_list = bits.generate_optimization_problem_bit_list(intervention)

    # step 2: initialization()
    # step 3: lower, upper = solve()
    pass


def generate_symbolic_objective_function():
    pass