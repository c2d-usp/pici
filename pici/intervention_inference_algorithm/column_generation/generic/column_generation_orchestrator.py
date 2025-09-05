
from itertools import product
import os
import sys
import copy
import logging

import gurobipy as gp
from gurobipy import GRB
from pandas import DataFrame
import networkx as nx

logger = logging.getLogger(__name__)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pici.graph.graph import Graph
from pici.graph.node import Node
from pici.intervention_inference_algorithm.column_generation.generic import bits
from pici.intervention_inference_algorithm.column_generation.generic.master_problem import (
    MasterProblem,
)
from pici.intervention_inference_algorithm.column_generation.generic.subproblem import (
    SubProblem,
)
from pici.intervention_inference_algorithm.linear_programming.linear_constraints import (
    calculate_constraints_empirical_probabilities,
    find_c_component_and_tail_set,
    get_c_component_in_reverse_topological_order,
    get_symbolical_constraints_probabilities_and_wc,
)
from pici.intervention_inference_algorithm.linear_programming.obj_function_generator import (
    ObjFunctionGenerator,
)
from pici.utils.scalable_graphs_helper import get_scalable_dataframe

from pici.intervention_inference_algorithm.column_generation.scalable_problem_init import (
    InitScalable,
)
from pici.utils._enum import ColumnGenerationParameters


class ColumnGenerationProblemOrchestrator:
    def __init__(
        self,
        dataFrame: DataFrame,
        dag: Graph,
        intervention: Node,
        target: Node,
        minimizes_objective_function: bool,
        
        parametric_columns: dict[str, tuple[list[int]]],
        betaVarsCost: list[float],
        betaVarsBitsX0: list[tuple[str]],
        betaVarsBitsX1: list[tuple[str]],
    ):

        self.intervention = intervention
        self.target = target
        self.dataFrame = dataFrame
        self.minimizes_objective_function = minimizes_objective_function

        self.topological_order: list[Node] = dag.topological_order

        objective_function = ObjFunctionGenerator(
            graph=dag,
            dataFrame=dataFrame,
            intervention=intervention,
            target=target,
        )

        intervention_latent_parent = objective_function.intervention.latent_parent
        c_component_endogenous_nodes = intervention_latent_parent.children
        considered_c_comp = list(
            (set(c_component_endogenous_nodes) & set(objective_function.considered_graph_nodes))
            | {objective_function.intervention}
        )

        self.reversed_ordered_considered_c_comp = get_c_component_in_reverse_topological_order(
            topo_order=self.topological_order,
            unob=intervention.latent_parent,
            considered_c_comp=considered_c_comp,
        )

        c_component_and_tail: list[Node] = find_c_component_and_tail_set(
            intervention.latent_parent, self.reversed_ordered_considered_c_comp
        )
        symbolical_constraints_probabilities, W = (
            get_symbolical_constraints_probabilities_and_wc(
                c_comp_order=self.reversed_ordered_considered_c_comp,
                c_component_and_tail=c_component_and_tail,
                topo_order=self.topological_order,
            )
        )
        # Seguinte loop é desnecessário?
        W_ordered = []
        for i, node in enumerate(dag.topological_order):
            if node in W:
                W_ordered.append(node)
        W_ordered.reverse()

        self.reversed_ordered_W_realizations = self._get_w_realizations(W_ordered)

        self.number_of_restrictions = len(W)

        self.update_parents_to_reversed_topological_order(W_ordered)

        self.duals = {}
        for i in range(self.number_of_restrictions):
            self.duals[i] = ColumnGenerationParameters.BIG_M.value

        self.symbolic_objective_function_probabilites: list[tuple] = (
            objective_function.generate_symbolic_objective_function_probabilities()
        )

        self.symbolic_decision_function: dict[tuple, int] = (
            objective_function.generate_symbolic_decision_function()
        )

        self.bits_list: list[int] = bits.generate_optimization_problem_bit_list(
            intervention
        )

        self.constraints_empirical_probabilities: list[float] = (
            calculate_constraints_empirical_probabilities(
                data=dataFrame,
                Wc=W,
                symbolical_constraints_probabilities=symbolical_constraints_probabilities,
            )
        )

        self.columns_base = None
        self.master = MasterProblem()

        # vvvvvv Bloco antigo vvvvvvvvvv
        N = 1
        M = 1
        self.amountBitsPerCluster = 1 << (M + 1)
        self.amountBetaVarsPerX = 1 << (M + N)

        # Order parametric_columns (XA1A2..AnB1...Bm)
        self.parametric_columns: dict[str, tuple[list[int]]] = parametric_columns
        self.betaVarsBitsX0 = betaVarsBitsX0
        self.betaVarsBitsX1 = betaVarsBitsX1
        self.betaVarsCost = betaVarsCost

        self.solution = {}
        self.subproblem = SubProblem(N=N, M=M)

    def update_parents_to_reversed_topological_order(self, node: Node) -> None:
        ordered_parents = []
        for node in self.topological_order:
            if node in node.parents:
                ordered_parents.append(node)
        ordered_parents.reverse()
        node.parents = ordered_parents

    def setup(self, method=1):
        """
        Sets up the master and subproblem models for column generation.

        Configures the Gurobi solving method, initializes the base columns, sets up the master problem with
        empirical probability constraints, and initializes the subproblem with all required parameters.

        Gurobi's methods (https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html#method)

        Args:
            method (int, optional): The Gurobi solving method to use. Defaults to 1 (barrier and dual simplex).
        """
        # Define gurobi running method
        self.master.model.setParam(GRB.Param.Method, method)
        self.subproblem.model.setParam(GRB.Param.Method, method)

        self.columns_base = self._generate_initial_column_base()
        self.master.setup(self.columns_base, self.constraints_empirical_probabilities)

        #### We're here
        self.subproblem.setup(
            reversed_ordered_considered_c_comp=self.reversed_ordered_considered_c_comp,
            reversed_ordered_W_realizations=self.reversed_ordered_W_realizations,
            amountBitsPerCluster=self.amountBitsPerCluster,
            amountBetaVarsPerX=self.amountBetaVarsPerX,
            duals=self.duals,
            amountNonTrivialRestrictions=self.number_of_restrictions,
            betaVarsCost=self.betaVarsCost,
            parametric_column=self.parametric_columns,
            betaVarsBitsX0=self.betaVarsBitsX0,
            betaVarsBitsX1=self.betaVarsBitsX1,
            N=self.N,
            M=self.M,
            interventionValue=self.intervention.value,
            minimizes_objective_function=self.minimizes_objective_function,
        )

    def _generate_initial_column_base(self) -> list[list[int]]:
        """
        Generate an initial base columns for the master problem as an identity matrix.

        This method creates an identity matrix of size (number_of_restrictions + 1) x (number_of_restrictions + 1),
        where each column corresponds to a basic feasible solution for the initial master problem.
        The resulting matrix is returned.

        Returns:
            list[list[int]]: The identity matrix.
        """
        columns_base: list[list[int]] = []
        for index in range(self.number_of_restrictions + 1):
            new_column = [0] * (self.number_of_restrictions + 1)
            new_column[index] = 1
            columns_base.append(new_column)
        return columns_base

    def exec(self) -> int:
        """
        Executes the column generation algorithm.

        Alternates between solving the master problem and the subproblem, adding new columns to the master problem
        until no columns with negative reduced cost are found or the maximum number of iterations is reached.

        Returns:
            int: The number of iterations performed.
        Raises:
            TimeoutError: If the maximum number of allowed iterations is exceeded.
        """
        iterations_counter = 0
        while True:
            self.master.model.optimize()
            if self.master.model.Status == gp.GRB.OPTIMAL:  # OPTIMAL
                b = self.master.model.objVal
                logger.info(f"--------->> Master solution found: {b}")
            elif self.master.model.Status == gp.GRB.USER_OBJ_LIMIT:
                b = self.master.model.objVal
                logger.info(
                    f"--------->> ColumnGenerationParameters.BIG_M.value Limit reached! Master solution found: {b}"
                )
            else:
                logger.error(
                    f"--------->>  Master solution not found. Gurobi status code: {self.master.model.Status}"
                )
            self.duals = self.master.model.getAttr("pi", self.master.constrs)
            logger.debug(f"Master Duals: {self.duals}")
            # self.master.model.write(f"master_{counter}.lp")
            self.subproblem.update(self.duals)
            self.subproblem.model.optimize()
            if self.subproblem.model.Status == gp.GRB.OPTIMAL:  # OPTIMAL
                b = self.subproblem.model.objVal
                logger.info(f"--------->> Subproblem solution found!: {b}")
            elif self.subproblem.model.Status == gp.GRB.USER_OBJ_LIMIT:
                b = self.subproblem.model.objVal
                logger.info(
                    f"--------->> ColumnGenerationParameters.BIG_M.value Limit reached! Subproblem solution found: {b}"
                )
            else:
                logger.error(
                    f"--------->>  Subproblem solution not found. Gurobi status code: {self.subproblem.model.Status}"
                )
            # self.subproblem.model.write(f"subproblem_{counter}.lp")

            reduced_cost = self.subproblem.model.objVal
            logger.debug(f"Reduced Cost: {reduced_cost}")
            if reduced_cost >= 0:
                break

            newColumn: list[int] = []
            for index in range(len(self.subproblem.bitsParametric)):
                newColumn.append(self.subproblem.bitsParametric[index].X)

            newColumn.append(
                1
            )  # For the equation sum(pi) = 1. This restriction is used in the MASTER problem.
            logger.debug(f"New Column: {newColumn}")

            objCoeff: float = 0.0
            for betaIndex in range(self.amountBetaVarsPerX):
                if self.intervention.value == 0:
                    objCoeff += (
                        self.betaVarsCost[betaIndex]
                        * self.subproblem.beta_varsX0[betaIndex].X
                    )
                else:
                    objCoeff += (
                        self.betaVarsCost[betaIndex]
                        * self.subproblem.beta_varsX1[betaIndex].X
                    )

            self.master.update(
                new_column=newColumn,
                index=len(self.columns_base),
                obj_coeff=objCoeff,
                minimizes_objective_function=self.minimizes_objective_function,
            )
            self.columns_base.append(newColumn)
            iterations_counter += 1
            if iterations_counter >= ColumnGenerationParameters.MAX_ITERACTIONS_ALLOWED.value:
                raise TimeoutError(
                    f"Too many iterations (MAX:{ColumnGenerationParameters.MAX_ITERACTIONS_ALLOWED.value})"
                )
            logger.info(f"Iteration Number = {iterations_counter}")

        return iterations_counter
    
    def optimize_master(self) -> float:
        """
        Optimizes the master problem with continuous variables and writes the model to disk.

        Probelm master handles continuous probabilities in the interval [0,1].

        Sets all master problem variables to continuous type, solves the master problem using Gurobi,
        and writes the model to both LP and MPS file formats for inspection or debugging.

        Returns:
            float: The objective value of the optimized master problem.
        """
        self.master.model.setAttr("vType", self.master.vars, GRB.CONTINUOUS)
        self.master.model.optimize()
        self.master.model.write("model.lp")
        self.master.model.write("model.mps")
        return self.master.model.ObjVal

def solve(problem: ColumnGenerationProblemOrchestrator, method=1) -> tuple[int, float]:
    """
    Solves the column generation problem using the BIG_M approach.

    Args:
        problem (ColumnGenerationProblemBuilder): The column generation problem instance.
        method (int, optional): The Gurobi solving method to use. Defaults to 1.

    Returns:
        tuple[int, float]: A tuple containing the final objective bound and the number of iterations performed.
    """
    problem.setup(method)
    number_of_iterations = problem.exec()
    bound = problem.optimize_master()
    return bound, number_of_iterations

def get_node_list_realizations(node_list: list[Node]) -> list[list]:
    ranges = [range(node.cardinality) for node in node_list]
    cartesian = product(*ranges)
    matrix = [[node.label for node in node_list]]
    matrix += [list(combo) for combo in cartesian]
    # idx_A = matrix[0].index(node.label)
    return matrix


def buildScalarProblem(
    M: int, N: int, interventionValue: int, targetValue: int, df, minimizes_objective_function: bool
):
    # Calculate the empirical probs (RHS of the restrictions, so b in Ax=b)
    empiricalProbabilities: list[float] = InitScalable.calculateEmpiricals(
        N=N, M=M, df=df
    )
    # Auxiliary Gamma U variables (beta): calculate the obj coeff in the subproblem and the relation to the bit variables that compose them
    betaVarsCoeffObjSubproblem: list[float] = []
    betaVarsBitsX0, betaVarsCoeffObjSubproblemX0 = (
        InitScalable.defineGammaUAuxiliaryVariables(
            M=M, N=N, df=df, targetValue=targetValue, XValue=0,
        )
    )
    betaVarsBitsX1, betaVarsCoeffObjSubproblemX1 = (
        InitScalable.defineGammaUAuxiliaryVariables(
            M=M, N=N, df=df, targetValue=targetValue, XValue=1,
        )
    )
    if interventionValue == 1:
        betaVarsCoeffObjSubproblem = copy.deepcopy(betaVarsCoeffObjSubproblemX1)
    else:
        betaVarsCoeffObjSubproblem = copy.deepcopy(betaVarsCoeffObjSubproblemX0)

    # Parametric_columns:
    parametric_columns: list[tuple[list[str]]] = InitScalable.defineParametricColumn(
        M=M, N=N
    )
    return ColumnGenerationProblemOrchestrator(
        dataFrame=df,
        empiricalProbabilities=empiricalProbabilities,
        parametric_columns=parametric_columns,
        N=N,
        M=M,
        betaVarsCost=betaVarsCoeffObjSubproblem,
        betaVarsBitsX0=betaVarsBitsX0,
        betaVarsBitsX1=betaVarsBitsX1,
        interventionValue=interventionValue,
        minimizes_objective_function=minimizes_objective_function,
    )


def exemplo_de_execucao():
    N = 1
    M = 2
    scalable_df = get_scalable_dataframe(M=M, N=N)
    interventionValue = 1
    targetValue = 1

    scalarProblem = buildScalarProblem(
        M=M,
        N=N,
        interventionValue=interventionValue,
        targetValue=targetValue,
        df=scalable_df,
        minimizes_objective_function=True,
    )
    lower, itLower = solve(scalarProblem)

    scalarProblem = buildScalarProblem(
        M=M,
        N=N,
        interventionValue=interventionValue,
        targetValue=targetValue,
        df=scalable_df,
        minimizes_objective_function=False,
    )
    upper, itUpper = solve(scalarProblem)
    upper = -upper
    logger.info(f"{lower} =< P(Y = {targetValue}|X = {interventionValue}) <= {upper}")
    logger.info(f"{itLower} iteracoes para lower e {itUpper} para upper")


def main():
    return exemplo_de_execucao()
