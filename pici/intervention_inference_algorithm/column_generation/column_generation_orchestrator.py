import logging
import os
import sys

import gurobipy as gp
from gurobipy import GRB
from pandas import DataFrame

THIS_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../.."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)
logger.debug("Logging configured to DEBUG for orchestrator and imported modules")

from pici.graph.graph import (
    Graph,
    order_list_in_reversed_topological_order,
    update_parents_to_reversed_topological_order,
)
from pici.graph.node import Node
from pici.intervention_inference_algorithm.column_generation.master_problem import (
    MasterProblem,
)
from pici.intervention_inference_algorithm.column_generation.subproblem import (
    SubProblem,
    get_node_list_realizations,
)
from pici.intervention_inference_algorithm.column_generation.colum_gen_constraints import (
    calculate_number_of_constraints,
    column_gen_calculate_constraints_empirical_probabilities,
    find_c_component_and_tail_set,
    get_c_component_in_reverse_topological_order,
    get_symbolical_constraints_probabilities_and_wc,
)
from pici.intervention_inference_algorithm.column_generation.column_gen_obj_function_generator import (
    ObjFunctionGenerator,
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
        minimizes_objective_function: bool,
    ):
        self.dag = dag
        self.intervention = intervention
        self.target = target
        self.dataFrame = dataFrame
        self.minimizes_objective_function = minimizes_objective_function
        self.duals = {}

        if dag.topological_order is None or len(dag.topological_order) == 0:
            raise Exception("dag.topological_order is None")
        self.topological_order: list[Node] = dag.topological_order

        objective_function = ObjFunctionGenerator(
            graph=dag,
            dataFrame=dataFrame,
            intervention=intervention,
            target=target,
        )

        self.symbolic_objective_function_probabilites: list[tuple] = (
            objective_function.generate_symbolic_objective_function_probabilities()
        )

        considered_graph_nodes = objective_function.considered_graph_nodes
        intervention_latent_parent = self.intervention.latent_parent
        c_component_endogenous_nodes = intervention_latent_parent.children
        considered_c_comp = list(
            (set(c_component_endogenous_nodes) & set(considered_graph_nodes))
            | {self.intervention}
        )

        self.reversed_ordered_considered_c_comp = (
            get_c_component_in_reverse_topological_order(
                topo_order=self.topological_order,
                unob=self.intervention.latent_parent,
                considered_c_comp=considered_c_comp,
            )
        )

        c_component_and_tail: list[Node] = find_c_component_and_tail_set(
            self.intervention.latent_parent, self.reversed_ordered_considered_c_comp
        )

        symbolical_constraints_probabilities, W = (
            get_symbolical_constraints_probabilities_and_wc(
                considered_c_comp_in_topo_order=self.reversed_ordered_considered_c_comp,
                c_component_and_tail=c_component_and_tail,
                topo_order=self.topological_order,
            )
        )

        W_ordered = []
        for node in self.topological_order:
            if node in W:
                W_ordered.append(node)

        W_ordered.reverse()
        self.reversed_ordered_W = W_ordered

        self.reversed_ordered_W_realizations = get_node_list_realizations(
            self.reversed_ordered_W
        )

        update_parents_to_reversed_topological_order(
            self.reversed_ordered_W, self.topological_order
        )

        self.constraints_empirical_probabilities: list[float] = (
            column_gen_calculate_constraints_empirical_probabilities(
                data=dataFrame,
                symbolical_constraints_probabilities=symbolical_constraints_probabilities,
                reversed_ordered_W_realizations=self.reversed_ordered_W_realizations,
            )
        )

        considered_c_comp_plus_adapted_tail = (
            self.get_considered_c_comp_plus_adapted_tail(
                self.reversed_ordered_considered_c_comp, self.intervention
            )
        )
        self.reversed_ordered_considered_c_comp_plus_adapted_tail = (
            order_list_in_reversed_topological_order(
                self.topological_order, considered_c_comp_plus_adapted_tail
            )
        )

        self.number_of_constraints = calculate_number_of_constraints(W=W)
        self.transposed_columns_base = None

    def get_considered_c_comp_plus_adapted_tail(
        self, reversed_ordered_considered_c_comp, intervention
    ):
        """
        Considered_c_comp_plus_adapted_tail = considered_c-comp + parents of every node in considered_c-comp except the parents of the intervention.
        """
        considered_c_comp_plus_adapted_tail = set()
        for node in reversed_ordered_considered_c_comp:
            considered_c_comp_plus_adapted_tail.add(node)
            if node != intervention:
                considered_c_comp_plus_adapted_tail.update(
                    [parent for parent in node.parents if not parent.is_latent]
                )
        return list(considered_c_comp_plus_adapted_tail)

    def solve(self, method=0) -> tuple[int, float]:
        """
        Solves the column generation problem using the BIG_M approach.

        Args:
            problem (ColumnGenerationProblemBuilder): The column generation problem instance.
            method (int, optional): The Gurobi solving method to use. Defaults to 1.

        Returns:
            tuple[int, float]: A tuple containing the final objective bound and the number of iterations performed.
        """
        self.setup(method)
        number_of_iterations = self.column_generation()
        bound = self.optimize_master()
        return bound, number_of_iterations

    def setup(self, method=0):
        """
        Sets up the master and subproblem models for column generation.

        Configures the Gurobi solving method, initializes the base columns, sets up the master problem with
        empirical probability constraints, and initializes the subproblem with all required parameters.

        Gurobi's methods (https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html#method)

        Args:
            method (int, optional): The Gurobi solving method to use. Defaults to 1 (barrier and dual simplex).
        """
        # Define gurobi running method
        self.master = MasterProblem()
        self.subproblem = SubProblem(
            df=self.dataFrame,
            intervention=self.intervention,
            target=self.target,
            minimizes_objective_function=self.minimizes_objective_function,
        )

        self.master.model.setParam(GRB.Param.Method, method)
        self.subproblem.model.setParam(GRB.Param.Method, method)

        self.transposed_columns_base = self._generate_initial_column_base(
            number_of_constraints=self.number_of_constraints
        )
        self.master.setup(
            self.transposed_columns_base, self.constraints_empirical_probabilities
        )

        self.subproblem.setup(
            reversed_ordered_considered_c_comp=self.reversed_ordered_considered_c_comp,
            reversed_ordered_W_realizations=self.reversed_ordered_W_realizations,
            reversed_ordered_W=self.reversed_ordered_W,
            symbolic_objective_function_probabilites=self.symbolic_objective_function_probabilites,
            reversed_ordered_considered_c_comp_plus_adapted_tail=self.reversed_ordered_considered_c_comp_plus_adapted_tail,
        )

    def _generate_initial_column_base(
        self, number_of_constraints: int
    ) -> list[list[int]]:
        """
        Generate an initial base columns for the master problem as an identity matrix.

        This method creates an identity matrix of size (number_of_constraints + 1) x (number_of_constraints + 1),
        where each column corresponds to a basic feasible solution for the initial master problem.
        The resulting matrix is returned.

        Returns:
            list[list[int]]: The identity matrix.
        """
        transposed_columns_base: list[list[int]] = []
        for index in range(number_of_constraints + 1):
            new_column = [0] * (number_of_constraints + 1)
            new_column[index] = 1
            transposed_columns_base.append(new_column)
        return transposed_columns_base

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
        already_added_columns = []
        data = []
        start = 0
        while True:
            self.master.model.optimize()
            if self.master.model.Status == gp.GRB.OPTIMAL:  # OPTIMAL
                b = self.master.model.objVal
                current_master_solution = b
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
            # self.master.model.write(f"sca_cgo_master_{iterations_counter}.lp")
            self.duals = {k: float(round(x)) for k, x in self.duals.items()}
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
            # self.subproblem.model.write(f"sca_cgo_subproblem_{iterations_counter}.lp")

            reduced_cost = self.subproblem.model.objVal
            logger.debug(f"Reduced Cost: {reduced_cost}")
            if reduced_cost >= 0:
                break
            new_column: list[int] = []
            for index in range(len(self.subproblem.parameterized_column)):
                new_column.append(self.subproblem.parameterized_column[index].X)

            # For the equation sum(pi) = 1. This restriction is used in the MASTER problem.
            new_column.append(1)
            new_column = [float(round(x)) for x in new_column]
            logger.debug(f"New Column: {new_column}")

            gamma_coef: float = 0.0
            for (
                bit_product,
                var_gurobi,
            ) in self.subproblem.gamma_u_map_bit_product_to_linearized_variable.items():
                str_bit = ""
                for bit in bit_product.bit_list:
                    str_bit += (
                        f"({bit.sign}*[{bit.gurobi_var.VarName}:{bit.gurobi_var.X}]), "
                    )
                gamma_coef += bit_product.coef * var_gurobi.X

            logger.debug(f"BitProduct List: {str_bit}")
            logger.debug(f"{iterations_counter} gamma_coef: {gamma_coef}")
            logger.debug(
                "-------------------------------------------------------------------------------"
            )
            logger.debug("Cluster Bits:")
            for (
                node_label,
                dict_parents_realization,
            ) in self.subproblem.cluster_bits.items():
                logger.debug(f"-- Node: {node_label}")
                for (
                    parents_realization_key,
                    cluster_gurobi_var,
                ) in dict_parents_realization.items():
                    logger.debug(f"---- Parents Realization: {parents_realization_key}")
                    for idx, var in cluster_gurobi_var.items():
                        logger.debug(
                            f"-------- {idx}-ith Gurobi Var Name: {var.VarName}"
                        )
            logger.debug(
                "-------------------------------------------------------------------------------"
            )
            if str(new_column) not in already_added_columns:
                self.master.update(
                    new_column=new_column,
                    index=len(self.transposed_columns_base),
                    obj_coeff=gamma_coef,
                    minimizes_objective_function=self.minimizes_objective_function,
                )
                self.transposed_columns_base.append(new_column)
                already_added_columns.append(str(new_column))
            else:
                print("Repeated column")
                print(f"--------->> Subproblem solution found!: {self.subproblem.model.objVal}")
                pass
            iterations_counter += 1
            if (
                iterations_counter
                >= ColumnGenerationParameters.MAX_ITERACTIONS_ALLOWED.value
            ):
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
        # self.master.model.write("sca_cgo_model.lp")
        # self.master.model.write("sca_cgo_model.mps")
        return self.master.model.ObjVal
