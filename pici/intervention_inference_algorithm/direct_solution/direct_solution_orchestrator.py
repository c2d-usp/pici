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

from pici.intervention_inference_algorithm.direct_solution.obj_function_generator import (
    ObjFunctionGenerator,
)
from pici.intervention_inference_algorithm.direct_solution.get_constraints import (
    calculate_number_of_constraints,
    calculate_constraints_empirical_probabilities,
    find_c_component_and_tail_set,
    get_c_component_in_reverse_topological_order,
    get_symbolical_constraints_probabilities_and_wc,
)
from pici.intervention_inference_algorithm.direct_solution.optimization_problem import (
    OptimizationProblem,
    get_node_list_realizations,
)


class DirectSolutionOrchestrator:
    def __init__(
        self,
        dataFrame: DataFrame,
        dag: Graph,
        intervention: Node,
        target: Node,
        minimizes_objective_function: bool,
        max_time=1200,
    ):
        self.dag = dag
        self.intervention = intervention
        self.target = target
        self.dataFrame = dataFrame
        self.minimizes_objective_function = minimizes_objective_function
        self.max_time = max_time

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
            calculate_constraints_empirical_probabilities(
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

    def solve(self, method=-1) -> tuple[int, float]:
        """
        Solves the column generation problem using the BIG_M approach.

        Args:
            problem (ColumnGenerationProblemBuilder): The column generation problem instance.
            method (int, optional): The Gurobi solving method to use. Defaults to 1.

        Returns:
            tuple[int, float]: A tuple containing the final objective bound and the number of iterations performed.
        """
        self.setup(method)
        self.opt_problem.model.optimize()

        if self.opt_problem.model.status == GRB.OPTIMAL:
            print("Optimal solution found!")
            # Proceed to get variable values
        else:
            logger.error(
                    f"--------->> Solution not found. Gurobi status code: {self.opt_problem.model.Status}"
                )
        bound = self.opt_problem.model.ObjVal
        return bound

    def setup(self, method=-1):
        """
        Sets up the master and subproblem models for column generation.

        Configures the Gurobi solving method, initializes the base columns, sets up the master problem with
        empirical probability constraints, and initializes the subproblem with all required parameters.

        Gurobi's methods (https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html#method)

        Args:
            method (int, optional): The Gurobi solving method to use. Defaults to 1 (barrier and dual simplex).
        """
        # Define gurobi running method
        self.opt_problem = OptimizationProblem(
            df=self.dataFrame,
            intervention=self.intervention,
            target=self.target,
            num_constraints=self.number_of_constraints,
            minimizes_objective_function=self.minimizes_objective_function,
            max_time=self.max_time
        )
        self.opt_problem.model.setParam(GRB.Param.Method, method)
        self.opt_problem.setup(
            self.constraints_empirical_probabilities,
            self.reversed_ordered_considered_c_comp,
            self.reversed_ordered_W_realizations,
            self.reversed_ordered_W,
            self.symbolic_objective_function_probabilites,
            self.reversed_ordered_considered_c_comp_plus_adapted_tail,
        )
