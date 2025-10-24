from venv import logger
import pandas as pd
from experiments.utils.scalable_graphs_helper import generate_three_latents_binary_scalable_cardinalities, generate_three_latents_scalable_string_edges, get_three_latents_scalable_dataframe
from pici.causal_model import CausalModel
from pici.intervention_inference_algorithm.column_generation.column_generation_orchestrator import ColumnGenerationProblemOrchestrator
from pici.utils._enum import DataExamplesPaths


def exemplo_discrete_balke():
    balke_input = "Z -> X, X -> Y, U1 -> X, U1 -> Y, U2 -> Z"
    balke_cardinalities = {"Z": 4, "X": 3, "Y": 2, "U1": 0, "U2": 0}
    balke_unobs = ["U1", "U2"]
    balke_target = "Y"
    balke_target_value = 1
    balke_intervention = "X"
    balke_intervention_value = 1
    balke_csv_path = DataExamplesPaths.CSV_DISCRETE_IV_RANDOM_EXAMPLE.value
    balke_df = pd.read_csv(balke_csv_path)

    balke_model = CausalModel(
        data=balke_df,
        edges=balke_input,
        custom_cardinalities=balke_cardinalities,
        unobservables_labels=balke_unobs,
        interventions=(balke_intervention, balke_intervention_value),
        target=(balke_target, balke_target_value),
    )
    dataFrame = balke_df
    dag = balke_model.graph
    intervention = balke_model.interventions[0]
    target = balke_model.target
    minimizes_objective_function = True
    problem = ColumnGenerationProblemOrchestrator(
        dataFrame, dag, intervention, target, minimizes_objective_function
    )
    min_bound, min_iter = problem.solve()
    # logger.info(f"{min_bound} <= P({target.label}={balke_target_value} | do({intervention.label}={balke_intervention_value}))")
    dataFrame = balke_df
    dag = balke_model.graph
    intervention = balke_model.interventions[0]
    target = balke_model.target
    problem = ColumnGenerationProblemOrchestrator(
        dataFrame, dag, intervention, target, minimizes_objective_function=False
    )
    max_bound, max_iter = problem.solve()
    # logger.info(f"P({target.label}={balke_target_value} | do({intervention.label}={balke_intervention_value})) <= {max_bound}")
    logger.info(
        f"{min_bound} <= P({target.label}={balke_target_value} | do({intervention.label}={balke_intervention_value})) <= {max_bound}"
    )


def exemplo_binary_balke():
    balke_input = "Z -> X, X -> Y, U1 -> X, U1 -> Y, U2 -> Z"
    balke_cardinalities = {"Z": 2, "X": 2, "Y": 2, "U1": 0, "U2": 0}
    balke_unobs = ["U1", "U2"]
    balke_target = "Y"
    balke_target_value = 1
    balke_intervention = "X"
    balke_intervention_value = 1
    balke_csv_path = DataExamplesPaths.CSV_BALKE_PEARL_EXAMPLE.value
    balke_df = pd.read_csv(balke_csv_path)

    balke_model = CausalModel(
        data=balke_df,
        edges=balke_input,
        custom_cardinalities=balke_cardinalities,
        unobservables_labels=balke_unobs,
        interventions=(balke_intervention, balke_intervention_value),
        target=(balke_target, balke_target_value),
    )
    dataFrame = balke_df
    dag = balke_model.graph
    intervention = balke_model.interventions[0]
    target = balke_model.target
    minimizes_objective_function = True
    problem = ColumnGenerationProblemOrchestrator(
        dataFrame, dag, intervention, target, minimizes_objective_function
    )
    min_bound, min_iter = problem.solve()

    dataFrame = balke_df
    dag = balke_model.graph
    intervention = balke_model.interventions[0]
    target = balke_model.target
    problem = ColumnGenerationProblemOrchestrator(
        dataFrame, dag, intervention, target, minimizes_objective_function=False
    )
    max_bound, max_iter = problem.solve()

    logger.info(
        f"{min_bound} <= P({target.label}={balke_target_value} | do({intervention.label}={balke_intervention_value})) <= {max_bound}"
    )


def exemplo_n1_m2():
    n1_m2_input = "X -> A1, X -> B1, X -> B2, B1 -> A1, B2 -> A1, A1 -> Y, U1 -> X, U1 -> A1, U2 -> B1, U2 -> B2, U2 -> Y"
    n1_m2_cardinalities = {"X": 2, "Y": 2, "B1": 2, "B2": 2, "A1": 2, "U1": 0, "U2": 0}

    # n2m1 n1_m2_input = "X -> A1, A1 -> A2, X -> B1, A2 -> Y, U1 -> X, U1 -> A1, U1 -> A2, U2 -> B1, U2 -> Y"
    # n2m1 n1_m2_cardinalities = {"X": 2, "Y": 2, "B1": 2, "A2": 2, "A1": 2, "U1": 0, "U2": 0}

    n1_m2_unobs = ["U1", "U2"]
    n1_m2_target = "Y"
    n1_m2_target_value = 1
    n1_m2_intervention = "X"
    n1_m2_intervention_value = 1
    n1_m2_csv_path = DataExamplesPaths.CSV_3_LATENTS_N1M2.value
    n1_m2_df = pd.read_csv(n1_m2_csv_path)

    n1_m2_model = CausalModel(
        data=n1_m2_df,
        edges=n1_m2_input,
        custom_cardinalities=n1_m2_cardinalities,
        unobservables_labels=n1_m2_unobs,
        interventions=(n1_m2_intervention, n1_m2_intervention_value),
        target=(n1_m2_target, n1_m2_target_value),
    )
    dataFrame = n1_m2_df
    dag = n1_m2_model.graph
    intervention = n1_m2_model.interventions[0]
    target = n1_m2_model.target
    minimizes_objective_function = True
    problem = ColumnGenerationProblemOrchestrator(
        dataFrame, dag, intervention, target, minimizes_objective_function
    )
    min_bound, min_iter = problem.solve()

    intervention = n1_m2_model.interventions[0]
    target = n1_m2_model.target
    problem = ColumnGenerationProblemOrchestrator(
        dataFrame, dag, intervention, target, minimizes_objective_function=False
    )
    max_bound, max_iter = problem.solve()

    logger.info(
        f"{min_bound} <= P({target.label}={n1_m2_target_value} | do({intervention.label}={n1_m2_intervention_value})) <= {max_bound}"
    )


def example_scalable_n_m(N, M):
    edges = generate_three_latents_scalable_string_edges(N=N, M=M)
    cardinalities = generate_three_latents_binary_scalable_cardinalities(N=N, M=M)
    unobs = ["U1", "U2"]

    target = "Y"
    target_value = 1
    intervention = "X"
    intervention_value = 1
    df = get_three_latents_scalable_dataframe(M=M, N=N)

    model = CausalModel(
        data=df,
        edges=edges,
        custom_cardinalities=cardinalities,
        unobservables_labels=unobs,
        interventions=(intervention, intervention_value),
        target=(target, target_value),
    )
    dataFrame = df
    dag = model.graph
    intervention = model.interventions[0]
    target = model.target
    problem = ColumnGenerationProblemOrchestrator(
        dataFrame=dataFrame,
        dag=dag,
        intervention=intervention,
        target=target,
        minimizes_objective_function=True)
    min_bound, min_iter = problem.solve()

    intervention = model.interventions[0]
    target = model.target
    problem = ColumnGenerationProblemOrchestrator(
        dataFrame,
        dag,
        intervention,
        target,
        minimizes_objective_function=False
    )
    max_bound, max_iter = problem.solve()

    logger.info(
        f"{min_bound} <= P({target.label}={target_value} | do({intervention.label}={intervention_value})) <= {max_bound}"
    )

if __name__ == "__main__":
    # exemplo_discrete_balke()
    # exemplo_binary_balke()
    # exemplo_n1_m2()
    example_scalable_n_m(N=2, M=1)
