

from experiments.three_latents_scalables.scalable_problem_column_gen import ScalarProblem
from experiments.utils.scalable_graphs_helper import find_true_value_in_three_latents_scalable_graphs, get_example_input, get_three_latents_scalable_dataframe
from pici.causal_model import CausalModel
from pici.intervention_inference_algorithm.column_generation.column_generation_orchestrator import ColumnGenerationProblemOrchestrator, solve


def linear(N, M, intervention_label="X", intervention_value=1, target_label="Y", target_value=1, case="three_latent"):

    edges, df, unobs = get_example_input(N, M, case)
    model = CausalModel(
        data=df,
        edges=edges,
        unobservables_labels=unobs,
        interventions=(intervention_label, intervention_value),
        target=(target_label, target_value),
    )
    lower, upper = (
        model.partially_identifiable_intervention_query()
    )
    return_dict = {}
    return_dict["result"]["min_val"] = lower
    return_dict["result"]["max_val"] = upper
    return_dict["result"]["min_iter"] = None
    return_dict["result"]["max_iter"] = None
    return return_dict

def ad_hoc_column_generation(N, M, intervention_value=1, target_value=1, case="three_latent"):
    _, df, _ = get_example_input(N, M, case)
    scalarProblem = ScalarProblem.buildScalarProblem(
        M=M,
        N=N,
        interventionValue=intervention_value,
        targetValue=target_value,
        df=df,
        minimum=True,
    )
    lower, lower_iterations = scalarProblem.solve()


    scalarProblem = ScalarProblem.buildScalarProblem(
        M=M,
        N=N,
        interventionValue=intervention_value,
        targetValue=target_value,
        df=df,
        minimum=False,
    )
    upper, upper_iterations = scalarProblem.solve()
    upper = -upper

    return_dict = {}
    return_dict["result"]["min_val"] = lower
    return_dict["result"]["max_val"] = upper
    return_dict["result"]["min_iter"] = lower_iterations
    return_dict["result"]["max_iter"] = upper_iterations
    return return_dict

def generic_column_generation(N, M, intervention_label="X", intervention_value=1, target_label="Y", target_value=1, case="three_latent"):

    edges, df, unobs = get_example_input(N, M, case)
    model = CausalModel(
        data=df,
        edges=edges,
        unobservables_labels=unobs,
        interventions=(intervention_label, intervention_value),
        target=(target_label, target_value),
    )
    problem = ColumnGenerationProblemOrchestrator(
        dataFrame=df,
        dag=model.graph,
        intervention=model.interventions[0],
        target=model.target,
        minimizes_objective_function=True)
    min_bound, min_iter = solve(problem)

    problem = ColumnGenerationProblemOrchestrator(
        dataFrame=df,
        dag=model.graph,
        intervention=model.interventions[0],
        target=model.target,
        minimizes_objective_function=False
    )
    max_bound, max_iter = solve(problem)

    return_dict = {}
    return_dict["result"]["min_val"] = min_bound
    return_dict["result"]["max_val"] = max_bound
    return_dict["result"]["min_iter"] = min_iter
    return_dict["result"]["max_iter"] = max_iter
    return return_dict

def ad_hoc_true_value(N, M, intervention_value=1, target_value=1):

    df = get_three_latents_scalable_dataframe(M=M, N=N)
    truth = find_true_value_in_three_latents_scalable_graphs(N, M, target_value, intervention_value, df)

    return_dict = {}
    return_dict["result"]["exact"] = truth
    return return_dict
