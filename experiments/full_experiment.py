
import time as tm
import logging

logger = logging.getLogger(__name__)

import pandas as pd
from itertools import product
from causal_reasoning.utils.probabilities_helper import find_conditional_probability2, find_probability2
from causal_reasoning.interventional_do_calculus_algorithm.scalable_problem_column_gen import ScalarProblem
from causal_reasoning.utils.get_scalable_df import getScalableDataFrame
from causal_reasoning.causal_model import CausalModel
from causal_reasoning.utils.data_gen import generate_data_for_scale_case

EXPERIMENT_PATH = "./outputs/second_full_flags_experiment_results.csv"
ERROR_PATH = "./outputs/second_full_flags_error_log.txt"


def run_test(N, M, method, presolve, numeric_focus, fea_tol, opt_tol, i):
    experiments_df = pd.read_csv(EXPERIMENT_PATH)

    new_row = {'N':N,'M':M,'GC_LOWER_BOUND':None,'GC_UPPER_BOUND':None,
               'GC_LOWER_BOUND_REQUIRED_ITERATIONS':None,'GC_UPPER_BOUND_REQUIRED_ITERATIONS':None,
               'GC_SECONDS_TAKEN':None, 'METHOD':method, 'PRESOLVE':presolve, 'NUMERIC_FOCUS':numeric_focus,
                 'FEASIBILITY_TOL':fea_tol, 'OPTIMALITY_TOL':opt_tol}
    new_row_df = pd.DataFrame([new_row])

    interventionValue = 1; targetValue = 1
    scalable_df = None
    try:
        scalable_df = getScalableDataFrame(M=M, N=N)
    except Exception as e:
        logger.error(f"SCALABLE DF Error_N:{N},M:{M}_METHOD:{method}_PRESOLVE:{presolve}_NUM_FOC:{numeric_focus}_FEASIBILITYTOL:{fea_tol}_OPTIMALITYTOL:{opt_tol}_: {e}")
        with open(ERROR_PATH, 'a') as file:
            file.write(f"SCALABLE DF Error for {i}th -- N:{N},M:{M}_METHOD:{method}_PRESOLVE:{presolve}_NUM_FOC:{numeric_focus}_FEASIBILITYTOL:{fea_tol}_OPTIMALITYTOL:{opt_tol}: {e}\n")
        return
    try:
        start = tm.time()
        scalarProblem = ScalarProblem.buildScalarProblem(M=M, N=N, interventionValue=interventionValue, targetValue=targetValue, df=scalable_df, minimum = True)
        logger.info("MIN Problem Built")
        lower , lower_iterations = scalarProblem.solve(method=method, presolve=presolve,numeric_focus=numeric_focus,opt_tol=opt_tol,fea_tol=fea_tol)
        logger.info(f"Minimum Optimization N:{N},M:{M}_METHOD:{method}_PRESOLVE:{presolve}_NUM_FOC:{numeric_focus}_FEASIBILITYTOL:{fea_tol}_OPTIMALITYTOL:{opt_tol}: Lower: {lower}, Iterations: {lower_iterations}")

        scalarProblem = ScalarProblem.buildScalarProblem(M=M, N=N, interventionValue=interventionValue, targetValue=targetValue, df=scalable_df, minimum = False)
        logger.info("MAX Problem Built")
        upper, upper_iterations = scalarProblem.solve(method=method, presolve=presolve,numeric_focus=numeric_focus,opt_tol=opt_tol,fea_tol=fea_tol)
        upper = -upper
        logger.info(f"Maximum Optimization N:{N},M:{M}_METHOD:{method}_PRESOLVE:{presolve}_NUM_FOC:{numeric_focus}_FEASIBILITYTOL:{fea_tol}_OPTIMALITYTOL:{opt_tol}: Upper: {upper}, Iterations: {upper_iterations}")
        end = tm.time()
        total_time = end-start
        new_row_df['GC_LOWER_BOUND'] = lower
        new_row_df['GC_UPPER_BOUND'] = upper
        new_row_df['GC_LOWER_BOUND_REQUIRED_ITERATIONS'] = lower_iterations 
        new_row_df['GC_UPPER_BOUND_REQUIRED_ITERATIONS'] = upper_iterations
        new_row_df['GC_SECONDS_TAKEN'] = total_time
        logger.info("GC Ran")
    except Exception as e:
        logger.error(f"GC Error_N:{N},M:{M}_METHOD:{method}_PRESOLVE:{presolve}_NUM_FOC:{numeric_focus}_FEASIBILITYTOL:{fea_tol}_OPTIMALITYTOL:{opt_tol}_: {e}")
        with open(ERROR_PATH, 'a') as file:
            file.write(f"GC Error for {i}th -- N:{N},M:{M}_METHOD:{method}_PRESOLVE:{presolve}_NUM_FOC:{numeric_focus}_FEASIBILITYTOL:{fea_tol}_OPTIMALITYTOL:{opt_tol}: {e}\n")   
    experiments_df = pd.concat([experiments_df, new_row_df], ignore_index=True)
    experiments_df.to_csv(EXPERIMENT_PATH, index=False)
    logger.info(f"CSV updated")



def main():
    logging.basicConfig(level=logging.INFO)

    df = pd.DataFrame(columns=['N','M','GC_LOWER_BOUND', 'GC_UPPER_BOUND', 'GC_LOWER_BOUND_REQUIRED_ITERATIONS','GC_UPPER_BOUND_REQUIRED_ITERATIONS', 'GC_SECONDS_TAKEN', 'METHOD'])
    df.to_csv(EXPERIMENT_PATH, index=False)
    N_M = [
        (3,1),(4,1),(5,1),(6,1),
        (3,2),(4,2),(5,2),
        (3,3),(4,3),
        (3,4),
        (2,5),
    ]

    # N_M = [
    #         (1,1),(1,2),(1,3),(1,4),(1,5),(1,6),
    #         (2,1),(2,2),(2,3),(2,4),(2,5),#(2,6),
    #         (3,1),(3,2),(3,3),(3,4),#(3,5),(3,6),
    #         (4,1),(4,2),(4,3),#(4,4),(4,5),(4,6),
    #         (5,1),(5,2),#(5,3),(5,4),(5,5),(5,6),
    #         (6,1),#(6,2),(6,3),(6,4),(6,5),(6,6),
    # ]
    METHOD = [0, 1, 2, 3, 4]
    PRESOLVE = [0, 1, 2]
    NUMERIC_FOCUS = [1, 2, 3]
    OPT_FEA_TOL = [1e-9, 1e-7, 1e-5, 1e-2]

    n_tests = 10
    for values in N_M:
        N, M = values
        for i in range(0, n_tests):
            for method in METHOD:
                for presolve in PRESOLVE:
                    for num_foc in NUMERIC_FOCUS:
                        for opt_tol in OPT_FEA_TOL:
                            for fea_tol in OPT_FEA_TOL:
                                logger.info(f"{i}th N:{N} M:{M}")
                                run_test(N=N, M=M, method=method, presolve=presolve, numeric_focus=num_foc, fea_tol=fea_tol, opt_tol=opt_tol, i=i)
    logger.info("Done")

if __name__=="__main__":
    main()
