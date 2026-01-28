import logging

import pandas as pd

logger = logging.getLogger(__name__)

from pici.intervention_inference_algorithm.direct_solution.direct_solution_orchestrator import DirectSolutionOrchestrator
from pici.graph.graph import Graph
from pici.graph.node import Node


def build_direct_solution_problem(
    graph: Graph,
    df: pd.DataFrame,
    intervention: Node,
    target: Node
) -> tuple[str, str]:

    problem = DirectSolutionOrchestrator(
        dataFrame=df, dag=graph, intervention=intervention, target=target, minimizes_objective_function=True
    )
    min_bound = problem.solve()
    
    problem = DirectSolutionOrchestrator(
        dataFrame=df, dag=graph, intervention=intervention, target=target, minimizes_objective_function=False
    )    
    max_bound = problem.solve()

    logger.info(
        f"Causal query: P({target.label}={target.intervened_value}|do({intervention.label}={intervention.intervened_value}))"
    )
    logger.info(f"Bounds: {min_bound} <= P <= {max_bound}")
    return min_bound, max_bound
