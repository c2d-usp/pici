from pici.causal_model import CausalModel
from pici.intervention_inference_algorithm.column_generation.generic.column_generation_orchestrator import ColumnGenerationProblemOrchestrator, solve
from pici.intervention_inference_algorithm.column_generation.scalable_problem_column_gen import ScalarProblem
from pici.utils.scalable_graphs_helper import generate_binary_scalable_cardinalities, generate_scalable_string_edges, get_scalable_dataframe


def example_scalable_n_m(N, M):
    edges = generate_scalable_string_edges(N=N, M=M)
    cardinalities = generate_binary_scalable_cardinalities(N=N, M=M)
    unobs = ["U1", "U2"]

    target = "Y"
    target_value = 1
    intervention = "X"
    intervention_value = 1
    df = get_scalable_dataframe(M=M, N=N)

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

def single_exec():
    i = 0
    for m in range(1,4):
        for n in range(1,6-i):
            print(f"Running M:{m}, N:{n}")
            try:
                example_scalable_n_m(N=n,M=m)
            except:
                continue
        i += 1
if __name__ == "__main__":
    single_exec()