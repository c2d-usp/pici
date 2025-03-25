from scipy.optimize import linprog

import pandas as pd

from causal_usp_icti.graph.graph import Graph
from causal_usp_icti.linear_algorithm.linear_constraints import generate_constraints
from causal_usp_icti.linear_algorithm.obj_function_generator import ObjFunctionGenerator


class OptProblemBuilder:
    def builder_linear_problem(
        graph: Graph,
        df: pd.DataFrame,
        intervention: str,
        intervention_value: int,
        target: str,
        target_value: int,
    ):
        objFG = ObjFunctionGenerator(
            graph=graph,
            dataFrame=df,
            intervention=graph.labelToIndex[intervention],
            intervention_value=intervention_value,
            target=graph.labelToIndex[target],
            target_value=target_value,
            empiricalProbabilitiesVariables=[],
            mechanismVariables=[],
            conditionalProbabilitiesVariables={},
            debugOrder=[],
        )
        objFG.find_linear_good_set()
        mechanisms = objFG.get_mechanisms_pruned()
        objFunctionCoefficients = objFG.build_objective_function(mechanisms)

        interventionLatentParent = objFG.graph.graphNodes[
            objFG.intervention
        ].latentParent
        cComponentEndogenous = objFG.graph.graphNodes[interventionLatentParent].children
        consideredEndogenousNodes = list(
            (set(cComponentEndogenous) & set(objFG.debugOrder)) | {objFG.intervention}
        )

        probs, decisionMatrix = generate_constraints(
            data=df,
            dag=objFG.graph,
            unob=objFG.graph.graphNodes[objFG.intervention].latentParent,
            consideredCcomp=consideredEndogenousNodes,
            mechanism=mechanisms,
        )

        print("-- DEBUG OBJ FUNCTION --")
        for i, coeff in enumerate(objFunctionCoefficients):
            print(f"c_{i} = {coeff}")

        print("-- DECISION MATRIX --")
        for i in range(len(decisionMatrix)):
            for j in range(len(decisionMatrix[i])):
                print(f"{decisionMatrix[i][j]} ", end="")
            print(f" = {probs[i]}")
        intervals = [(0, 1) for _ in range(len(decisionMatrix[0]))]
        lowerBoundSol = linprog(
            c=objFunctionCoefficients,
            A_ub=None,
            b_ub=None,
            A_eq=decisionMatrix,
            b_eq=probs,
            method="highs",
            bounds=intervals,
        )
        lowerBound = lowerBoundSol.fun

        upperBoundSol = linprog(
            c=[-x for x in objFunctionCoefficients],
            A_ub=None,
            b_ub=None,
            A_eq=decisionMatrix,
            b_eq=probs,
            method="highs",
            bounds=intervals,
        )

        upperBound = -upperBoundSol.fun

        print(
            f"Causal query: P({target}={target_value}|do({intervention}={intervention_value}))"
        )
        print(f"Bounds: {lowerBound} <= P <= {upperBound}")
