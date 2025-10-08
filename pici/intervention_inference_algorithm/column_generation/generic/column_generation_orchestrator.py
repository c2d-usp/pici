
from itertools import product
import os
import sys
import copy
import logging
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from pandas import DataFrame
import networkx as nx

from pici.causal_model import CausalModel
# THIS_DIR = os.getcwd()
# PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../.."))
# import sys
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pici.graph.graph import Graph, order_list_in_reversed_topological_order
from pici.graph.node import Node
from pici.intervention_inference_algorithm.column_generation.generic import bits
from pici.intervention_inference_algorithm.column_generation.generic.master_problem import (
    MasterProblem,
)
from pici.intervention_inference_algorithm.column_generation.generic.subproblem import (
    SubProblem, get_node_list_realizations
)
from pici.intervention_inference_algorithm.linear_programming.linear_constraints import (
    calculate_constraints_empirical_probabilities,
    calculate_number_of_constraints,
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
from pici.utils._enum import ColumnGenerationParameters, DataExamplesPaths

BIG_M = 1e4


class ColumnGenerationProblemOrchestrator:
    def __init__(
        self,
        dataFrame: DataFrame,
        dag: Graph,
        intervention: Node,
        target: Node,
        minimizes_objective_function: bool
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
                considered_c_comp_in_topo_order=self.reversed_ordered_considered_c_comp,
                c_component_and_tail=c_component_and_tail,
                topo_order=self.topological_order,
            )
        )

        if W is None:
            raise Exception("W is None")
        
        if dag.topological_order is None or len(dag.topological_order) == 0:
            raise Exception("dag.topological_order is None")
        
        W_ordered = []
        for i, node in enumerate(dag.topological_order):
            if node in W:
                W_ordered.append(node)

        W_ordered.reverse()
        self.reversed_ordered_W = W_ordered

        if self.reversed_ordered_W is not None:
            self.reversed_ordered_W_realizations = get_node_list_realizations(self.reversed_ordered_W)
        else:
            raise Exception("reversed is None")

        self.number_of_constraints = calculate_number_of_constraints(W=W)        

        # TODO: COLOCAR ISSO NA CONSTRUÇÃO DO OBJETO GRAPH
        self.update_parents_to_reversed_topological_order(self.reversed_ordered_W)

        self.duals = {}
        for i in range(self.number_of_constraints):
            self.duals[i] = ColumnGenerationParameters.BIG_M.value

        self.symbolic_objective_function_probabilites: list[tuple] = (
            objective_function.generate_symbolic_objective_function_probabilities()
        )
        '''
        NÃO USADAS. FORAM CRIADAS ANTES DE INICIAR A CONSTRUÇÃO DO CG.
        self.symbolic_decision_function: dict[tuple, int] = (
            objective_function.generate_symbolic_decision_function()
        )

        self.bits_list: list[int] = bits.generate_optimization_problem_bit_list(
            intervention
        )
        '''

        self.constraints_empirical_probabilities: list[float] = (
            calculate_constraints_empirical_probabilities(
                data=dataFrame,
                Wc=W,
                symbolical_constraints_probabilities=symbolical_constraints_probabilities,
            )
        )
        self.columns_base = None
        self.master = MasterProblem()
        self.subproblem = SubProblem(df=dataFrame, intervention=intervention, target=target)
    
    def get_conjunto_estranho(self, reversed_ordered_considered_c_comp, intervention):
        '''
        Conjunto estranho é o considered_c-comp + os pais de (considered_c-comp - X)
        
        '''
        conjunto_estranho = set()
        for node in reversed_ordered_considered_c_comp:
            conjunto_estranho.add(node)
            if node != intervention:
                conjunto_estranho.update([parent for parent in node.parents if not parent.is_latent])
        return list(conjunto_estranho)

    def update_parents_to_reversed_topological_order(self, node_list: list[Node]) -> None:
        for node in node_list:
            ordered_parents = []
            for ordered_node in self.topological_order:
                if ordered_node in node.parents:
                    ordered_parents.append(ordered_node)
            ordered_parents.reverse()
            node.parents = ordered_parents
        return node_list

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

        conjunto_estranho = self.get_conjunto_estranho(self.reversed_ordered_considered_c_comp, self.intervention)
        reversed_ordered_conjunto_estranho = order_list_in_reversed_topological_order(self.topological_order, conjunto_estranho)

        self.subproblem.setup(
            reversed_ordered_considered_c_comp=self.reversed_ordered_considered_c_comp,
            reversed_ordered_W_realizations=self.reversed_ordered_W_realizations,
            reversed_ordered_W=self.reversed_ordered_W,
            symbolic_objective_function_probabilites=self.symbolic_objective_function_probabilites,
            conjunto_estranho=reversed_ordered_conjunto_estranho,
            number_of_constraints=self.number_of_constraints,
            duals=self.duals,
        )

    def _generate_initial_column_base(self) -> list[list[int]]:
        """
        Generate an initial base columns for the master problem as an identity matrix.

        This method creates an identity matrix of size (number_of_constraints + 1) x (number_of_constraints + 1),
        where each column corresponds to a basic feasible solution for the initial master problem.
        The resulting matrix is returned.

        Returns:
            list[list[int]]: The identity matrix.
        """
        columns_base: list[list[int]] = []
        for index in range(self.number_of_constraints + 1):
            new_column = [0] * (self.number_of_constraints + 1)
            new_column[index] = 1
            columns_base.append(new_column)
        return columns_base

    def column_generation(self) -> int:
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
            # logger.debug(f"Master Duals: {self.duals}")
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
            for index in range(len(self.subproblem.coluna_parametrizada)):
                newColumn.append(self.subproblem.coluna_parametrizada[index].X)

            # For the equation sum(pi) = 1. This restriction is used in the MASTER problem.
            newColumn.append(1)
            logger.debug(f"New Column: {newColumn}")

            gamma_coef: float = 0.0
            for bit_product, var_gurobi in self.subproblem.gamma_u_map_bit_product_to_linearized_variable.items():
                str_bit = ""
                for bit in bit_product.bit_list:
                    str_bit += f"({bit.sign}*{bit.gurobi_var.VarName}), "
                gamma_coef += bit_product.coef * var_gurobi.X
            print(f"gamma_coef: {gamma_coef}")
            print(f"BitProduct List: {str_bit}")

            self.master.update(
                new_column=newColumn,
                index=len(self.columns_base),
                obj_coeff=gamma_coef,
                minimizes_objective_function=self.minimizes_objective_function,
            )
            self.columns_base.append(newColumn)

            print(f"len(columns_base) [Linhas] = {len(self.columns_base)}")
            print(f"len(columns_base[0]) [Colunas] = {len(self.columns_base[0])}")

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
    number_of_iterations = problem.column_generation()
    bound = problem.optimize_master()
    return bound, number_of_iterations

def exemplo_balke():
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
    minimizes_objective_function = 1
    problem = ColumnGenerationProblemOrchestrator(
        dataFrame,
        dag,
        intervention,
        target,
        minimizes_objective_function)
    solve(problem)

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
    n1_m2_csv_path = DataExamplesPaths.CSV_N1M2.value
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
    minimizes_objective_function = 1
    problem = ColumnGenerationProblemOrchestrator(
        dataFrame,
        dag,
        intervention,
        target,
        minimizes_objective_function)
    solve(problem)


if __name__ == '__main__': 
    exemplo_balke()
    # exemplo_n1_m2()
