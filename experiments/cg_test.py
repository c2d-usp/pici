import logging

from pici.causal_model import CausalModel
from pici.intervention_inference_algorithm.column_generation.column_generation_orchestrator import ColumnGenerationProblemOrchestrator, solve
from experiments.three_latents_scalables.scalable_problem_column_gen import ScalarProblem
from experiments.utils.scalable_graphs_helper import find_true_value_in_three_latents_scalable_graphs, generate_three_latents_binary_scalable_cardinalities, generate_three_latents_scalable_string_edges, get_three_latents_scalable_dataframe

logger = logging.getLogger(__name__)

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
    min_bound, min_iter = solve(problem)

    intervention = model.interventions[0]
    target = model.target
    problem = ColumnGenerationProblemOrchestrator(
        dataFrame,
        dag,
        intervention,
        target,
        minimizes_objective_function=False
    )
    max_bound, max_iter = solve(problem)

    scalarProblem = ScalarProblem.buildScalarProblem(
        M=M,
        N=N,
        interventionValue=intervention_value,
        targetValue=target_value,
        df=df,
        minimum=False,
    )
    upper, itUpper = scalarProblem.solve()
    upper = -upper
    scalarProblem = ScalarProblem.buildScalarProblem(
        M=M,
        N=N,
        interventionValue=intervention_value,
        targetValue=target_value,
        df=df,
        minimum=True,
    )
    lower, itLower = scalarProblem.solve()
    with open("cg_results.txt", "a") as f:
        f.write(f"--M:{M}, N:{N}\n")
        f.write(f"    Generic CG:\n")
        f.write(f"        {min_bound} <= P({target.label}={target_value} | do({intervention.label}={intervention_value})) <= {max_bound}\n")
        f.write(f"        {min_iter} iteracoes para lower e {max_iter} para upper\n")
        f.write(f"    Scalable CG:\n")
        f.write(f"        {lower} =< P(Y = {target_value}|X = {intervention_value}) <= {upper}\n")
        f.write(f"        {itLower} iteracoes para lower e {itUpper} para upper\n\n")
    # print(f"M:{M}, N:{N}")
    # print(f"    Generic CG:")
    # print(f"        {min_bound} <= P({target.label}={target_value} | do({intervention.label}={intervention_value})) <= {max_bound}")
    # print(f"        {min_iter} iteracoes para lower e {max_iter} para upper")
    # print(f"    Scalable CG:")
    # print(f"        {lower} =< P(Y = {target_value}|X = {intervention_value}) <= {upper}")
    # print(f"        {itLower} iteracoes para lower e {itUpper} para upper")


def scalable(N, M, intervention_value, target_value, df):
    try:
        scalarProblem = ScalarProblem.buildScalarProblem(
            M=M,
            N=N,
            interventionValue=intervention_value,
            targetValue=target_value,
            df=df,
            minimum=False,
        )
        upper, itUpper = scalarProblem.solve()
        upper = -upper
        scalarProblem = ScalarProblem.buildScalarProblem(
            M=M,
            N=N,
            interventionValue=intervention_value,
            targetValue=target_value,
            df=df,
            minimum=True,
        )
        lower, itLower = scalarProblem.solve()
        return lower, itLower, upper, itUpper
    except:
        return None, None, None, None

def generic(df, edges, cardinalities, unobs, intervention, target, intervention_value, target_value):
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
    try:
        problem = ColumnGenerationProblemOrchestrator(
            dataFrame=dataFrame,
            dag=dag,
            intervention=intervention,
            target=target,
            minimizes_objective_function=True)
        min_bound, min_iter = solve(problem)

        intervention = model.interventions[0]
        target = model.target
        problem = ColumnGenerationProblemOrchestrator(
            dataFrame,
            dag,
            intervention,
            target,
            minimizes_objective_function=False
        )
        max_bound, max_iter = solve(problem)
        return min_bound, min_iter, max_bound, max_iter
    except:
        return None, None, None, None

def single_exec():
    unobs = ["U1", "U2"]
    target = "Y"
    target_value = 1
    intervention = "X"
    intervention_value = 1
    i = 0
    for m in range(1,4):
        for n in range(1,6-i):
            print(f"Running M:{m}, N:{n}")
            edges = generate_three_latents_scalable_string_edges(N=n, M=m)
            cardinalities = generate_three_latents_binary_scalable_cardinalities(N=n, M=m)
            df = get_three_latents_scalable_dataframe(M=m, N=n)
            lower, itLower, upper, itUpper = scalable(n,m, intervention_value, target_value, df)
            min_bound, min_iter, max_bound, max_iter = generic(df, edges, cardinalities, unobs, intervention, target, intervention_value, target_value)
            with open("cg_results.txt", "a") as f:
                f.write("_____________________________________________________\n")
                f.write(f"M:{m}, N:{n}\n")
                f.write(f"    Generic CG:\n")
                f.write(f"        {min_bound} <= P({target}={target_value} | do({intervention}={intervention_value})) <= {max_bound}\n")
                f.write(f"        {min_iter} iteracoes para lower e {max_iter} para upper\n")
                f.write(f"    Scalable CG:\n")
                f.write(f"        {lower} =< P(Y = {target_value}|X = {intervention_value}) <= {upper}\n")
                f.write(f"        {itLower} iteracoes para lower e {itUpper} para upper\n\n")
        i += 1
if __name__ == "__main__":
    single_exec()


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
    minimizes_objective_function = True
    problem = ColumnGenerationProblemOrchestrator(
        dataFrame=dataFrame,
        dag=dag,
        intervention=intervention,
        target=target,
        minimizes_objective_function=True)
    min_bound, min_iter = solve(problem)

    intervention = model.interventions[0]
    target = model.target
    problem = ColumnGenerationProblemOrchestrator(
        dataFrame,
        dag,
        intervention,
        target,
        minimizes_objective_function=False
    )
    max_bound, max_iter = solve(problem)

    logger.debug(f"{min_bound} <= P({target.label}={target_value} | do({intervention.label}={intervention_value})) <= {max_bound}")
    logger.debug(f"True value: {find_true_value_in_three_latents_scalable_graphs(N=N, M=M, y0=1, x0=1,df=df)}")