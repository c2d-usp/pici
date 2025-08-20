
import os
import sys
import copy
import logging

import gurobipy as gp
from gurobipy import GRB

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

BIG_M = 1e4
DBG = False
MAX_ITERACTIONS_ALLOWED = 2000


class ColumnGenerationProblemBuilder:
    def __init__(
        self,
        dataFrame,
        parametric_columns: dict[str, tuple[list[int]]],
        betaVarsCost: list[float],
        betaVarsBitsX0: list[tuple[str]],
        betaVarsBitsX1: list[tuple[str]],
        dag: Graph,
        intervention: Node,
        target: Node,
        minimizes_objective_function: bool,
    ):

        self.intervention = intervention
        self.target = target
        self.dataFrame = dataFrame
        self.minimizes_objective_function = minimizes_objective_function

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

        topological_order: list[Node] = dag.topological_order

        c_comp_order = get_c_component_in_reverse_topological_order(
            topo_order=topological_order,
            unob=intervention.latent_parent,
            considered_c_comp=considered_c_comp,
        )
        c_component_and_tail: list[Node] = find_c_component_and_tail_set(
            intervention.latent_parent, c_comp_order
        )
        symbolical_constraints_probabilities, W = (
            get_symbolical_constraints_probabilities_and_wc(
                c_comp_order=c_comp_order,
                c_component_and_tail=c_component_and_tail,
                topo_order=topological_order,
            )
        )

        self.number_of_restrictions = len(W)

        self.duals = {}
        for i in range(self.number_of_restrictions):
            self.duals[i] = BIG_M

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

    def setup(self, method=1):
        # Define gurobi running method
        self.master.model.params.Method = method
        self.subproblem.model.params.Method = method

        self._initialize_column_base()
        self.master.setup(self.columns_base, self.constraints_empirical_probabilities)

        #### We're here
        self.subproblem.setup(
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

    def _initialize_column_base(self):
        # Initialize big-M problem with the identity block of size
        # equal to the amount of restrictions.
        columns_base: list[list[int]] = []
        for index in range(self.number_of_restrictions + 1):
            new_column = [0] * (self.number_of_restrictions + 1)
            new_column[index] = 1
            columns_base.append(new_column)
        self.columns_base = columns_base

    def exec(self):
        counter = 0
        while True:
            self.master.model.optimize()
            if self.master.model.Status == gp.GRB.OPTIMAL:  # OPTIMAL
                b = self.master.model.objVal
                logger.info(f"--------->> Master solution found: {b}")
            elif self.master.model.Status == gp.GRB.USER_OBJ_LIMIT:
                b = self.master.model.objVal
                logger.info(
                    f"--------->> BIG_M Limit reached! Master solution found: {b}"
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
                    f"--------->> BIG_M Limit reached! Subproblem solution found: {b}"
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
            counter += 1
            if counter >= MAX_ITERACTIONS_ALLOWED:
                raise TimeoutError(
                    f"Too many iterations (MAX:{MAX_ITERACTIONS_ALLOWED})"
                )
            logger.info(f"Iteration Number = {counter}")

        return counter
    
    def optimize_master(self):
        self.master.model.setAttr("vType", self.master.vars, GRB.CONTINUOUS)
        self.master.model.optimize()
        self.master.model.write("model.lp")
        self.master.model.write("model.mps")
        return self.master.model.ObjVal

def solve(problem: ColumnGenerationProblemBuilder, method=1) -> tuple[int, float]:
    """
    Gurobi does not support branch-and-price, as this requires to add columns
    at local nodes of the search tree. A heuristic is used instead, where the
    integrality constraints for the variables of the final root LP relaxation
    are installed and the resulting MIP is solved. Note that the optimal
    solution could be overlooked, as additional columns are not generated at
    the local nodes of the search tree.
    """
    problem.setup(method)
    numberIterations = problem.exec()
    bound = problem.optimize_master()
    return bound, numberIterations


def buildScalarProblem(
    M: int, N: int, interventionValue: int, targetValue: int, df, minimizes_objective_function: bool
):
    # Calculate the empirical probs (RHS of the restrictions, so b in Ax=b)
    empiricalProbabilities: list[float] = InitScalable.calculateEmpiricals(
        N=N, M=M, df=df, DBG=DBG
    )
    # Auxiliary Gamma U variables (beta): calculate the obj coeff in the subproblem and the relation to the bit variables that compose them
    betaVarsCoeffObjSubproblem: list[float] = []
    betaVarsBitsX0, betaVarsCoeffObjSubproblemX0 = (
        InitScalable.defineGammaUAuxiliaryVariables(
            M=M, N=N, df=df, targetValue=targetValue, XValue=0, DBG=DBG
        )
    )
    betaVarsBitsX1, betaVarsCoeffObjSubproblemX1 = (
        InitScalable.defineGammaUAuxiliaryVariables(
            M=M, N=N, df=df, targetValue=targetValue, XValue=1, DBG=DBG
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
    return ColumnGenerationProblemBuilder(
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
