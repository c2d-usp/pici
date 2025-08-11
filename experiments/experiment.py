import logging
import time as tm

logger = logging.getLogger(__name__)

from itertools import product

import pandas as pd

from pici.causal_model import CausalModel
from pici.do_calculus_algorithm.column_generation.scalable_problem_column_gen import (
    ScalarProblem,
)
from pici.utils.probabilities_helper import (
    find_conditional_probability2,
    find_probability2,
)

from pici.utils.scalable_graphs_helper import get_scalable_dataframe

GC_EXPERIMENT_PATH = "./outputs/gc_experiment_results.csv"
LP_EXPERIMENT_PATH = "./outputs/lp_experiment_results.csv"
TRUE_VALUE_EXPERIMENT_PATH = "./outputs/true_value_experiment_results.csv"
ERROR_PATH = "./outputs/error_log.txt"


def genGraph(N, M):
    scalable_input: str = "U1 -> X, U3 -> Y, "
    for i in range(1, N + 1):
        scalable_input += f"U1 -> A{i}, "
        if i == 1:
            scalable_input += "X -> A1, "
        else:
            scalable_input += f"A{i-1} -> A{i}, "
    scalable_input += f"A{N} -> Y, "

    for i in range(1, M + 1):
        scalable_input += f"U2 -> B{i}, "
        scalable_input += f"X -> B{i}, "
        for j in range(1, N + 1):
            scalable_input += f"B{i} -> A{j}, "

    return scalable_input[:-2]


def true_value(N, M, y0, x0, df):
    prob = 0
    for rlt in list(product([0, 1], repeat=2)):
        term = 1
        term *= find_conditional_probability2(
            dataFrame=df,
            targetRealization={"Y": y0},
            conditionRealization={f"A{N}": rlt[0]},
        )
        term *= find_conditional_probability2(
            dataFrame=df,
            targetRealization={f"A{N}": rlt[0]},
            conditionRealization={"U1": rlt[1], "X": x0},
        )
        term *= find_probability2(dataFrame=df, realizationDict={"U1": rlt[1]})
        prob += term
    return prob

def column_gen_call(N_M, intervention_value=1, target_value=1):
    for values in N_M:
        N, M = values
        for i in range(1, 6):
            logger.info(f"{i}th N:{N} M:{M}")
            experiments_df = pd.read_csv(GC_EXPERIMENT_PATH)
            new_row = {
                "N": N,
                "M": M,
                "GC_LOWER_BOUND": None,
                "GC_UPPER_BOUND": None,
                "GC_LOWER_BOUND_REQUIRED_ITERATIONS": None,
                "GC_UPPER_BOUND_REQUIRED_ITERATIONS": None,
                "GC_SECONDS_TAKEN": None,
                "LP_LOWER_BOUND": None,
                "LP_UPPER_BOUND": None,
                "LP_SECONDS_TAKEN": None,
                "TRUE_VALUE": None,
            }
            new_row_df = pd.DataFrame([new_row])

            scalable_df = None
            try:
                scalable_df = get_scalable_dataframe(M=M, N=N)
            except Exception as e:
                logger.error(f"SCALABLE DF Error_N:{N}_M:{M}_: {e}")
                with open(ERROR_PATH, "a") as file:
                    file.write(f"SCALABLE DF Error for {i}th -- N:{N},M:{M}: {e}\n")
            try:
                start = tm.time()
                scalarProblem = ScalarProblem.buildScalarProblem(
                    M=M,
                    N=N,
                    interventionValue=intervention_value,
                    targetValue=target_value,
                    df=scalable_df,
                    minimum=True,
                )
                logger.info("MIN Problem Built")
                lower, lower_iterations = scalarProblem.solve()
                logger.info(
                    f"Minimum Optimization N:{N}, M:{M}: Lower: {lower}, Iterations: {lower_iterations}"
                )

                scalarProblem = ScalarProblem.buildScalarProblem(
                    M=M,
                    N=N,
                    interventionValue=intervention_value,
                    targetValue=target_value,
                    df=scalable_df,
                    minimum=False,
                )
                logger.info("MAX Problem Built")
                upper, upper_iterations = scalarProblem.solve()
                upper = -upper
                logger.info(
                    f"Maximum Optimization N:{N}, M:{M}: Upper: {upper}, Iterations: {upper_iterations}"
                )
                end = tm.time()
                total_time = end - start
                new_row_df["GC_LOWER_BOUND"] = lower
                new_row_df["GC_UPPER_BOUND"] = upper
                new_row_df["GC_LOWER_BOUND_REQUIRED_ITERATIONS"] = lower_iterations
                new_row_df["GC_UPPER_BOUND_REQUIRED_ITERATIONS"] = upper_iterations
                new_row_df["GC_SECONDS_TAKEN"] = total_time
                logger.info("GC Ran")
            except Exception as e:
                logger.error(f"GC Error_N:{N}_M:{M}_: {e}")
                with open(ERROR_PATH, "a") as file:
                    file.write(f"GC Error for {i}th -- N:{N},M:{M}: {e}\n")

            experiments_df = pd.concat([experiments_df, new_row_df], ignore_index=True)
            experiments_df.to_csv(GC_EXPERIMENT_PATH, index=False)
            logger.info("COLUMN GENERATION: CSV updated")
    logger.info("COLUMN GENERATION: Done")

def lp_call(N_M, intervention_value=1, target_value=1):
    scalable_unobs = ["U1", "U2", "U3"]
    scalable_target = "Y"
    scalable_intervention = "X"

    for values in N_M:
        N, M = values
        scalable_input = genGraph(N, M)

        for i in range(1, 6):
            logger.info(f"{i}th N:{N} M:{M}")

            experiments_df = pd.read_csv(LP_EXPERIMENT_PATH)

            new_row = {
                "N": N,
                "M": M,
                "GC_LOWER_BOUND": None,
                "GC_UPPER_BOUND": None,
                "GC_LOWER_BOUND_REQUIRED_ITERATIONS": None,
                "GC_UPPER_BOUND_REQUIRED_ITERATIONS": None,
                "GC_SECONDS_TAKEN": None,
                "LP_LOWER_BOUND": None,
                "LP_UPPER_BOUND": None,
                "LP_SECONDS_TAKEN": None,
                "TRUE_VALUE": None,
            }
            new_row_df = pd.DataFrame([new_row])
            scalable_df = None
            try:
                scalable_df = get_scalable_dataframe(M=M, N=N)
            except Exception as e:
                logger.error(f"SCALABLE DF Error_N:{N}_M:{M}_: {e}")
                with open(ERROR_PATH, "a") as file:
                    file.write(f"SCALABLE DF Error for {i}th -- N:{N},M:{M}: {e}\n")
            try:
                start = tm.time()
                scalable_model = CausalModel(
                    data=scalable_df,
                    edges=scalable_input,
                    unobservables=scalable_unobs,
                    interventions=scalable_intervention,
                    interventions_value=intervention_value,
                    target=scalable_target,
                    target_value=target_value,
                )
                lower, upper = scalable_model.partially_identifiable_intervention_query()
                end = tm.time()
                total_time = end-start
                new_row_df["LP_LOWER_BOUND"] = lower
                new_row_df["LP_UPPER_BOUND"] = upper
                new_row_df["LP_SECONDS_TAKEN"] = total_time
                logger.info("LP Ran")
            except Exception as e:
                logger.error(f"LP Error_N:{N}_M:{M}_: {e}")
                with open(ERROR_PATH, "a") as file:
                    file.write(f"LP Error for {i}th -- N:{N},M:{M}: {e}\n")

            experiments_df = pd.concat([experiments_df, new_row_df], ignore_index=True)
            experiments_df.to_csv(LP_EXPERIMENT_PATH, index=False)
            logger.info("LP: CSV updated")
    logger.info("LP: Done")

def true_value_call(N_M, intervention_value=1, target_value=1):
    for values in N_M:
        N, M = values
        logger.info(f"TRUE VALUE: N:{N} M:{M}")

        experiments_df = pd.read_csv(TRUE_VALUE_EXPERIMENT_PATH)

        new_row = {
            "N": N,
            "M": M,
            "GC_LOWER_BOUND": None,
            "GC_UPPER_BOUND": None,
            "GC_LOWER_BOUND_REQUIRED_ITERATIONS": None,
            "GC_UPPER_BOUND_REQUIRED_ITERATIONS": None,
            "GC_SECONDS_TAKEN": None,
            "LP_LOWER_BOUND": None,
            "LP_UPPER_BOUND": None,
            "LP_SECONDS_TAKEN": None,
            "TRUE_VALUE": None,
        }
        new_row_df = pd.DataFrame([new_row])
        scalable_df = None
        try:
            scalable_df = get_scalable_dataframe(M=M, N=N)
        except Exception as e:
            logger.error(f"TRUE VALUE:SCALABLE DF Error_N:{N}_M:{M}_: {e}")
            with open(ERROR_PATH, "a") as file:
                file.write(f"TRUE VALUE:SCALABLE DF Error for N:{N},M:{M}: {e}\n")
        try:
            new_row_df["TRUE_VALUE"] = true_value(
                N, M, target_value, intervention_value, scalable_df
            )
            logger.info("TRUE VALUE Ran")
        except Exception as e:
            logger.error(f"True Value Error_N:{N}_M:{M}_: {e}")
            with open(ERROR_PATH, "a") as file:
                file.write(f"True value Error for N:{N},M:{M}: {e}\n")

        experiments_df = pd.concat([experiments_df, new_row_df], ignore_index=True)
        experiments_df.to_csv(TRUE_VALUE_EXPERIMENT_PATH, index=False)
        logger.info("TRUE VALUE: CSV updated")
    logger.info("TRUE VALUE: Done")

def main():
    logging.basicConfig(level=logging.INFO)

    df = pd.DataFrame(
        columns=[
            "N",
            "M",
            "GC_LOWER_BOUND",
            "GC_UPPER_BOUND",
            "GC_LOWER_BOUND_REQUIRED_ITERATIONS",
            "GC_UPPER_BOUND_REQUIRED_ITERATIONS",
            "GC_SECONDS_TAKEN",
            "LP_LOWER_BOUND",
            "LP_UPPER_BOUND",
            "LP_SECONDS_TAKEN",
            "TRUE_VALUE",
        ]
    )
    df.to_csv(GC_EXPERIMENT_PATH, index=False)
    df.to_csv(LP_EXPERIMENT_PATH, index=False)
    df.to_csv(TRUE_VALUE_EXPERIMENT_PATH, index=False)

    N_M = [
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 1),
        (1, 2),
        (2, 2),
        (3, 2),
        (4, 2),
        (1, 3),
        (2, 3),
        (1, 4),
    ]

    true_value(N_M)
    column_gen_call(N_M)
    lp_call(N_M)

if __name__ == "__main__":
    main()
