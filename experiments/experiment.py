
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

def genGraph(N, M):
    scalable_input: str = "U1 -> X, U3 -> Y, "
    for i in range(1,N + 1):
        scalable_input += f"U1 -> A{i}, "
        if (i == 1):
            scalable_input += "X -> A1, "
        else:
            scalable_input += f"A{i-1} -> A{i}, "
    scalable_input += f"A{N} -> Y, "

    for i in range(1,M + 1):
        scalable_input += f"U2 -> B{i}, "
        scalable_input += f"X -> B{i}, "
        for j in range(1,N + 1):
            scalable_input += f"B{i} -> A{j}, "
            
    return scalable_input[:-2]


def true_value(N,M,y0,x0,df):
    prob = 0 
    for rlt in list(product([0, 1], repeat= 2)):
        term = 1
        term *= find_conditional_probability2(dataFrame=df,targetRealization={"Y":y0},conditionRealization={f"A{N}": rlt[0]})
        term *= find_conditional_probability2(dataFrame=df,targetRealization={f"A{N}": rlt[0]},conditionRealization= {"U1": rlt[1], "X": x0})
        term *= find_probability2(dataFrame= df, realizationDict={"U1":rlt[1]})
        prob += term
    return prob

def main():
    logging.basicConfig(level=logging.INFO)

    df = pd.DataFrame(columns=['N','M','GC_LOWER_BOUND', 'GC_UPPER_BOUND', 'GC_LOWER_BOUND_REQUIRED_ITERATIONS','GC_UPPER_BOUND_REQUIRED_ITERATIONS', 'GC_SECONDS_TAKEN', 'LP_LOWER_BOUND', 'LP_UPPER_BOUND', 'LP_SECONDS_TAKEN', 'TRUE_VALUE'])
    df.to_csv("./outputs/experiment_results.csv", index=False)

    scalable_unobs = ["U1", "U2", "U3"]
    scalable_target = "Y"; target_value = 1
    scalable_intervention = "X"; intervention_value = 1   
    N_M = [(1,4)]#, (5, 1)]#(1,1),(2,1),(1,2),(3,1),(4,1),(2,2),(1,3),(2,3), (5,1), (1,4),(4,2)]

    # -1=automatic,

    # 0=primal simplex,

    # 1=dual simplex,

    # 2=barrier,

    # 3=concurrent,

    # 4=deterministic concurrent, and
    methods = [-1, 0, 1, 2, 3, 4]
    n_tests = 1
    for values in N_M:
        N, M = values
        # scalable_input = genGraph(N, M)
        for method in methods:
            for i in range(0, n_tests):
                logger.info(f"{i}th N:{N} M:{M}")

                experiments_df = pd.read_csv("./outputs/experiment_results.csv")
        
                new_row = {'N':N,'M':M,'GC_LOWER_BOUND':None,'GC_UPPER_BOUND':None,'GC_LOWER_BOUND_REQUIRED_ITERATIONS':None,'GC_UPPER_BOUND_REQUIRED_ITERATIONS':None, 'GC_SECONDS_TAKEN':None, 'LP_LOWER_BOUND':None, 'LP_UPPER_BOUND':None, 'LP_SECONDS_TAKEN':None, 'TRUE_VALUE':None, 'METHOD':method}
                new_row_df = pd.DataFrame([new_row])

                #generate_data_for_scale_case(n=N, m=M)

                interventionValue = 1; targetValue = 1
                scalable_df = None
                try:
                    scalable_df = getScalableDataFrame(M=M, N=N)
                except Exception as e:
                    logger.error(f"SCALABLE DF Error_N:{N}_M:{M}_: {e}")
                    with open("./outputs/error_log.txt", 'a') as file:
                        file.write(f"SCALABLE DF Error for {i}th -- N:{N},M:{M}: {e}\n")
                try:
                    start = tm.time()
                    scalarProblem = ScalarProblem.buildScalarProblem(M=M, N=N, interventionValue=interventionValue, targetValue=targetValue, df=scalable_df, minimum = True)
                    logger.info("MIN Problem Built")
                    lower , lower_iterations = scalarProblem.solve(method=method)
                    logger.info(f"Minimum Optimization N:{N}, M:{M}: Lower: {lower}, Iterations: {lower_iterations}")

                    scalarProblem = ScalarProblem.buildScalarProblem(M=M, N=N, interventionValue=interventionValue, targetValue=targetValue, df=scalable_df, minimum = False)
                    logger.info("MAX Problem Built")
                    upper, upper_iterations = scalarProblem.solve(method=method)
                    upper = -upper
                    logger.info(f"Maximum Optimization N:{N}, M:{M}: Upper: {upper}, Iterations: {upper_iterations}")
                    end = tm.time()
                    total_time = end-start
                    new_row_df['GC_LOWER_BOUND'] = lower
                    new_row_df['GC_UPPER_BOUND'] = upper
                    new_row_df['GC_LOWER_BOUND_REQUIRED_ITERATIONS'] = lower_iterations 
                    new_row_df['GC_UPPER_BOUND_REQUIRED_ITERATIONS'] = upper_iterations
                    new_row_df['GC_SECONDS_TAKEN'] = total_time
                    logger.info("GC Ran")
                except Exception as e:
                    logger.error(f"GC Error_N:{N}_M:{M}_: {e}")
                    with open("./outputs/error_log.txt", 'a') as file:
                        file.write(f"GC Error for {i}th -- N:{N},M:{M}: {e}\n")

                # if N>=5 and M>=5:
                #     try:
                #         start = tm.time()
                #         scalable_model = CausalModel(
                #             data=scalable_df,
                #             edges=scalable_input,
                #             unobservables=scalable_unobs,
                #             interventions=scalable_intervention,
                #             interventions_value=intervention_value,
                #             target=scalable_target,
                #             target_value=target_value,
                #         )
                #         lower, upper = scalable_model.inference_query(gurobi=True)
                #         end = tm.time()
                #         new_row_df['LP_LOWER_BOUND'] = lower
                #         new_row_df['LP_UPPER_BOUND'] = upper
                #         new_row_df['LP_SECONDS_TAKEN'] = total_time
                #         logger.info("LP Ran")
                #     except Exception as e:
                #         logger.error(f"LP Error_N:{N}_M:{M}_: {e}")
                #         with open("./outputs/error_log.txt", 'a') as file:
                #             file.write(f"LP Error for {i}th -- N:{N},M:{M}: {e}\n")
                
                # try:
                #     new_row_df['TRUE_VALUE'] = true_value(N,M,target_value,intervention_value,scalable_df)
                #     logger.info("TRUE VALUE  Ran")
                # except Exception as e:
                #     logger.error(f"True Value Error_N:{N}_M:{M}_: {e}")
                #     with open("./outputs/error_log.txt", 'a') as file:
                #         file.write(f"True value Error for {i}th -- N:{N},M:{M}: {e}\n")           

                experiments_df = pd.concat([experiments_df, new_row_df], ignore_index=True)
                experiments_df.to_csv("./outputs/experiment_results.csv", index=False)
                logger.info(f"CSV updated")
    logger.info("Done")

if __name__=="__main__":
    main()
