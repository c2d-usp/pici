import logging

import pandas as pd

logger = logging.getLogger(__name__)

from pici.intervention_inference_algorithm.column_generation.column_generation_orchestrator import ColumnGenerationProblemOrchestrator
from pici.graph.graph import Graph
from pici.graph.node import Node


def build_column_generation_problem(
    graph: Graph,
    df: pd.DataFrame,
    intervention: Node,
    target: Node,
    gurobi_params: dict,
    column_gen_max_iter: int,
) -> tuple[str, str]:
    problem = ColumnGenerationProblemOrchestrator(
        dataFrame=df, 
        dag=graph, 
        intervention=intervention, 
        target=target, 
        minimizes_objective_function=True, 
        gurobi_params=gurobi_params,
        column_gen_max_iter=column_gen_max_iter,
    )
    min_bound, min_iter = problem.solve()
    
    problem = ColumnGenerationProblemOrchestrator(
        dataFrame=df,
        dag=graph, 
        intervention=intervention, 
        target=target, 
        minimizes_objective_function=False, 
        gurobi_params=gurobi_params,
        column_gen_max_iter=column_gen_max_iter,
    )    
    max_bound, max_iter = problem.solve()

    logger.info(
        f"Causal query: P({target.label}={target.intervened_value}|do({intervention.label}={intervention.intervened_value}))"
    )
    logger.info(f"Bounds: {min_bound} <= P <= {max_bound}")
    return min_bound, max_bound
