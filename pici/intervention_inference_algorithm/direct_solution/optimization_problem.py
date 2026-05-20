import logging
import os
import sys
import math
from itertools import product

import gurobipy as gp
from gurobipy import GRB, Var, tupledict
from pandas import DataFrame

from pici.intervention_inference_algorithm.direct_solution.bits import (
    Bit,
    BitProduct,
)
from pici.utils._enum import (
    GurobiParameters,
)

from pici.graph.node import Node
from pici.utils.probabilities_helper import find_conditional_probability

logger = logging.getLogger(__name__)

THIS_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../.."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.path.append(os.path.abspath(os.path.join(THIS_DIR, PROJECT_ROOT)))


class OptimizationProblem:
    def __init__(self,
            intervention: Node,
            target: Node,
            num_constraints: int,
            df: DataFrame = None,
            minimizes_objective_function=False,
            max_time=1200,
        ):
        self.model = gp.Model("opt_problem")
        self.intervention = intervention
        self.target = target
        self.df = df
        self.num_constraints = num_constraints
        self.cluster_bits: list[dict[str, dict[str, tupledict[int, Var]]]] = []
        self.minimizes_objective_function = minimizes_objective_function
        self.vars = None
        self.max_time = max_time

    def setup(
        self,
        constraints_empirical_probabilities: list[float],
        reversed_ordered_considered_c_comp: list[Node],
        reversed_ordered_W_realizations: list[list],
        reversed_ordered_W: list[Node],
        symbolic_objective_function_probabilites: list[tuple],
        reversed_ordered_considered_c_comp_plus_adapted_tail: list[Node],
    ):
        """
        Initializes the optimization problem with base columns and empirical probability constraints.

        Args:
            transposed_columns_base (list[list[int]]): The base columns is an identity matrix for the initial variables.
            constraints_empirical_probabilities (list[float]): The right-hand side values for the empirical probability constraints.

        This method creates variables for each base column, sets up the constraints so that the
        linear combination of columns matches the empirical probabilities, and configures the model
        for minimization. Gurobi output is suppressed for iterative procedures.
        """
        if self.minimizes_objective_function:
            model_sense = GRB.MINIMIZE
        else:
            model_sense = GRB.MAXIMIZE
        
        self.parameterized_columns = []

        self.reversed_ordered_W = reversed_ordered_W
        self.reversed_ordered_considered_c_comp = reversed_ordered_considered_c_comp
        self.reversed_ordered_W_realizations = reversed_ordered_W_realizations

        self._create_cluster_bits(reversed_ordered_considered_c_comp)
        self._add_constraints_cluster_bits(reversed_ordered_considered_c_comp)
        self.vars = self.model.addVars(
            self.num_constraints,
            vtype=GRB.CONTINUOUS,
            name="P_u",
        )

        # Define objetive function in terms of bits

        realization_reversed_ordered_considered_c_comp_plus_adapted_tail = (
            get_node_list_realizations(
                reversed_ordered_considered_c_comp_plus_adapted_tail
            )
        )

        self.objective_function_vars_not_in_W = (
            self.get_objective_function_vars_not_in_W(
                symbolic_objective_function_probabilites, reversed_ordered_W
            )
        )

        self.Pw, self.Pq = self.separate_objective_function_probabilities(
            symbolic_objective_function_probabilites, reversed_ordered_W
        )
        self.realization_objective_function_vars_not_in_W = get_node_list_realizations(
            self.objective_function_vars_not_in_W
        )

        
        self.gamma_u_map_bit_product_to_linearized_variable: list[dict[BitProduct, Var]] = (
            self.gamma_linearize(
                reversed_ordered_considered_c_comp,
                realization_reversed_ordered_considered_c_comp_plus_adapted_tail,
                self.realization_objective_function_vars_not_in_W,
            )
        )
        self.model.update()
        gamma_coefs = []
        for k in range(self.num_constraints):
            self.generate_linearized_bit_products_constraints(
                self.gamma_u_map_bit_product_to_linearized_variable[k], name=f"gamma_{k}"
            )
            gamma_coef = 0.0
            for (
                    bit_product,
                    var_gurobi,
                ) in self.gamma_u_map_bit_product_to_linearized_variable[k].items():
                    gamma_coef += bit_product.coef * var_gurobi
            gamma_coefs.append(gamma_coef)

        self.model.setObjective(
            gp.quicksum(
                gamma_coefs[k] * self.vars[k]
                for k in range(self.num_constraints)
                ),
            model_sense
        )
        self.a_u_map_bit_product_to_linearized_variable: list[dict[BitProduct, Var]] = []

        self.get_A_u_column_and_parameterized_column(
            reversed_ordered_W_realizations, reversed_ordered_considered_c_comp
        )

        self.alphas = list()

        for row in range(len(constraints_empirical_probabilities)):
            alpha_row = []
            for column in range(self.num_constraints):
                alpha = self.model.addVar(
                    vtype=GRB.CONTINUOUS, name=f"alpha_{row}{column}"
                )
                alpha_row.append(alpha)

            self.alphas.append(alpha_row)

        for k in range(self.num_constraints):
            self.generate_linearize_alphas_constraints(
                k, self.a_u_map_bit_product_to_linearized_variable[k], name=f"au_{k}"
            )        

        self.model.addConstrs(
            (
                gp.quicksum(
                    self.alphas[realization_id][column_id]
                    for column_id in range(self.num_constraints)
               )
                == constraints_empirical_probabilities[realization_id]
                for realization_id in range(len(constraints_empirical_probabilities))
            ),
           name="EmpiricalRestrictions",
        )

        #self.model.setParam(GRB.Param.OutputFlag, 0)
        self.model.setParam(GRB.Param.TimeLimit, self.max_time)
        self.model.setParam(GRB.Param.MIPGap, 0.1)
        self.model.setParam(GRB.Param.MIPFocus, 3)
        self.model.setParam(GRB.Param.VarBranch, 2)
        self.model.setParam(GRB.Param.PreQLinearize, 1)
        self.model.setParam(GRB.Param.Presolve, 2)
        self.model.setParam(GRB.Param.PreSparsify, 2)
        self.model.setParam(GRB.Param.ScaleFlag, 2)
        self.model.setParam(GRB.Param.NormAdjust, 1)
        self.model.setParam(GRB.Param.BranchDir, -1)
        self.model.setParam(GRB.Param.FeasibilityTol, 1e-3)
        self.model.setParam(GRB.Param.DegenMoves, 0)
        self.model.setParam(GRB.Param.Heuristics, 1)
        self.model.setParam(GRB.Param.Cuts, 3)
        self.model.setParam(GRB.Param.RINS, 100)
        self.model.setParam(GRB.Param.Aggregate, 0)
        self.model.setParam(GRB.Param.InfProofCuts, 0)
        self.model.update()
    
    def _create_cluster_bits(self, considered_c_comp: list[Node]):
        """
        Each node in the considered c-component has a series of bits that represents each realization.
        Example:
            A = b0b1b2
            B = b0
            C = b0b1
            We've three clusters. Cluster A with 3 bits, Cluster B with one bit, and Cluster C with two bits.

        """
        cluster_bits_counter = 0
        logger.debug(f"Cluster Bits of :{considered_c_comp}")
        for k in range(self.num_constraints):
            self.cluster_bits.append({})
            for node in considered_c_comp:
                parents_without_latent = [
                    parent for parent in node.parents if not parent.is_latent
                ]
                reversed_ordered_node_parents_realizations: list[list] = (
                    get_node_list_realizations(parents_without_latent)
                )
                header = reversed_ordered_node_parents_realizations[0]
                logger.debug(f"Parents: {header}")
                reversed_ordered_node_parents_realizations = (
                    reversed_ordered_node_parents_realizations[1:]
                )
                node_number_of_bits = math.ceil(math.log2(node.cardinality))
                self.cluster_bits[k][node.label] = {}
                for i, realization in enumerate(reversed_ordered_node_parents_realizations):
                    realization_key: str = self.get_realization_key(header, realization)
                    logger.debug(
                        f"    {cluster_bits_counter}th - Node {node.label} Realization key: {realization_key}--{realization} ==> {node_number_of_bits} bits"
                    )
                    cluster_bits_counter += 1
                    self.cluster_bits[k][node.label][realization_key] = self.model.addVars(
                        node_number_of_bits,
                        vtype=GRB.BINARY,
                        name=f"bit_realization_{i}th_of_node_{node.label}_{realization_key}_{k}",
                    )
    
    def gamma_linearize(
        self,
        reversed_ordered_considered_c_comp: list[Node],
        realization_reversed_ordered_considered_c_comp_plus_adapted_tail: list,
        realization_objective_function_vars_not_in_W: list,
    ) -> dict:
        """
        Generate Gamma U: gamma_u_map_bit_product_to_linearized_variable
        Maps the Bit Product and its coefficient into a linearization variable.
        """
        gamma_u_map_bit_product_to_linearized_variable: list[dict[BitProduct, Var]] = []
        header = realization_reversed_ordered_considered_c_comp_plus_adapted_tail[0]
        cartesian_products = (
            realization_reversed_ordered_considered_c_comp_plus_adapted_tail[1:]
        )

        for k in range(self.num_constraints):
            gamma_u_map_bit_product_to_linearized_variable.append({})
            for realization in cartesian_products:
                if (
                    self.target.label in header
                    and realization[header.index(self.target.label)]
                    != self.target.intervened_value
                ):
                    continue
                if (
                    realization[header.index(self.intervention.label)]
                    != self.intervention.intervened_value
                ):
                    continue

                coef = self.get_coef_from_objective_function(
                    header, realization, realization_objective_function_vars_not_in_W
                )
                bit_product: BitProduct = self.generate_bit_product(
                    k,
                    node_list=reversed_ordered_considered_c_comp,
                    header=header,
                    realization=realization,
                    consider_intervention=True,
                )
                bit_product.set_coef(coef)
                gamma_u_map_bit_product_to_linearized_variable[k][bit_product] = (
                    self.model.addVar(
                        vtype=GRB.BINARY, name=f"linearization_auxiliary_variable_{k}"
                    )
                )
        return gamma_u_map_bit_product_to_linearized_variable
    
    def get_objective_function_vars_not_in_W(
        self, symbolic_objective_function_probabilites, W
    ) -> list[Node]:
        objective_function_vars_not_in_W = set()
        for conditional_probability in symbolic_objective_function_probabilites:
            probability_target, conditioned_nodes = conditional_probability
            if probability_target not in W:
                objective_function_vars_not_in_W.add(probability_target)
            for node in conditioned_nodes:
                if node not in W:
                    objective_function_vars_not_in_W.add(node)
        return [node for node in objective_function_vars_not_in_W]
    
    def separate_objective_function_probabilities(
        self, symbolic_objective_function_probabilites, W
    ) -> tuple[list[tuple], list[tuple]]:
        """
        Pw: it's the set of all conditional probabilities in the Objective Function that every variables is in the W set.
        Pq:  it's the set of all conditional probabilities in the Objective Function except the ones in Pw
        """
        P_Q = []
        P_W = []
        for conditional_probability in symbolic_objective_function_probabilites:
            probability_target, conditioned_nodes = conditional_probability
            if probability_target in W and all(node in W for node in conditioned_nodes):
                P_W.append(conditional_probability)
                continue
            P_Q.append(conditional_probability)
        return (P_W, P_Q)
    
    def get_coef_from_objective_function(
        self,
        w_header: list,
        w_realization: list,
        realization_objective_function_vars_not_in_W: list,
    ):
        coefw = 1
        logger.debug("Coeficients")
        logger.debug("W")
        logger.debug(f"{w_header}")
        logger.debug(f"{w_realization}")
        for w_conditional_probability in self.Pw:
            w_target, w_conditioned = w_conditional_probability

            w_target.value = w_realization[w_header.index(w_target.label)]
            str_w = ""
            for node in w_conditioned:
                node.value = w_realization[w_header.index(node.label)]
                str_w += f"{node.label}={node.value}, "
            curr = find_conditional_probability(
                dataFrame=self.df,
                target_realization=[w_target],
                condition_realization=w_conditioned,
            )
            logger.debug(
                f"P({w_target.label}={w_target.value}|{str_w[:len(str_w)-2]}) == {curr}"
            )

            coefw *= curr
        # ToDo: rename this var
        if len(self.objective_function_vars_not_in_W) <= 0:
            return coefw

        coefq = 0
        q_header = realization_objective_function_vars_not_in_W[0]
        q_realizations = realization_objective_function_vars_not_in_W[1:]
        logger.debug("_____________________________________")
        logger.debug("Q")
        for q_realization in q_realizations:
            if (
                self.intervention.label in q_header
                and q_realization[q_header.index(self.intervention.label)]
                != self.intervention.intervened_value
            ):
                continue
            if (
                self.target.label in q_header
                and q_realization[q_header.index(self.target.label)]
                != self.target.intervened_value
            ):
                continue
            logger.debug("--------------------")
            logger.debug(f"{q_header}")
            logger.debug(f"{q_realization}")
            partial_coef = 1
            for q_conditional_probability in self.Pq:
                q_target, q_conditioned = q_conditional_probability

                if q_target in self.reversed_ordered_W:
                    q_target.value = w_realization[w_header.index(q_target.label)]
                else:
                    q_target.value = q_realization[q_header.index(q_target.label)]

                str_q = ""
                for node in q_conditioned:
                    if node in self.reversed_ordered_W:
                        node.value = w_realization[w_header.index(node.label)]
                    else:
                        node.value = q_realization[q_header.index(node.label)]
                    str_q += f"{node.label}={node.value}, "
                curr = find_conditional_probability(
                    dataFrame=self.df,
                    target_realization=[q_target],
                    condition_realization=q_conditioned,
                )
                partial_coef *= curr
                logger.debug(
                    f"P({q_target.label}={q_target.value}|{str_q[:len(str_q)-2]}) == {curr}"
                )
            logger.debug(f"Coef_Parcial: {partial_coef}")
            coefq += partial_coef
            logger.debug(f"coefW: {coefw}")
            logger.debug(f"coefq: {coefq}")
            logger.debug(f"coef: {coefw*coefq}")
            logger.debug("----")
        return coefq * coefw
    
    def _add_constraints_cluster_bits(self, considered_c_comp: list[Node]):
        """
        Each node in the considered c-component has a series of bits that represents each realization.
        """
        cluster_bits_counter = 0
        logger.debug(f"Constraints fot the Cluster Bits of :{considered_c_comp}")
        for i in range(len(self.cluster_bits)):
            for node in considered_c_comp:
                parents_without_latent = [
                    parent for parent in node.parents if not parent.is_latent
                ]
                reversed_ordered_node_parents_realizations: list[list] = (
                    get_node_list_realizations(parents_without_latent)
                )
                header = reversed_ordered_node_parents_realizations[0]
                logger.debug(f"{header}")
                reversed_ordered_node_parents_realizations = (
                    reversed_ordered_node_parents_realizations[1:]
                )
                node_number_of_bits = math.ceil(math.log2(node.cardinality))

                for realization in reversed_ordered_node_parents_realizations:
                    realization_key: str = self.get_realization_key(header, realization)
                    logger.debug(
                        f"    {cluster_bits_counter}th - Constraint of Node {node.label} Realization key: {realization_key}--{realization}"
                    )
                    cluster_bits_counter += 1
                    debug_expr_str = ""
                    expr = 0
                    for variable_index in range(node_number_of_bits):
                        expr += (2 ** (variable_index)) * self.cluster_bits[i][node.label][
                            realization_key
                        ][variable_index]
                        debug_expr_str += f"2^{variable_index} * b{variable_index} + "
                    debug_expr_str += f"<= {node.cardinality-1}"
                    logger.debug(debug_expr_str)
                    self.model.addConstr(
                        expr <= node.cardinality - 1,
                        name=f"discrete_constraint_of_node_{node.label}_{realization_key}_{i}",
                    )

    def get_realization_key(self, header: list[str], realization: list[int]) -> str:
        if len(header) != len(realization):
            raise ValueError(
                "Lists with different sizes. Header and Realization should have the same size."
            )
        realization_key = ""
        for i in range(len(header)):
            realization_key += f"{header[i]}={realization[i]},"
        return realization_key[: len(realization_key) - 1]
    
    def get_A_u_column_and_parameterized_column(
        self, total_w_realization, considered_c_component_in_topological_order
    ):
        header = total_w_realization[0]
        total_w_realization = total_w_realization[1:]
        for k in range(self.num_constraints):
            parameterized_column = {}
            self.a_u_map_bit_product_to_linearized_variable.append({})
            for i, realization in enumerate(total_w_realization):
                bit_product: BitProduct = self.generate_bit_product(
                    k,
                    node_list=considered_c_component_in_topological_order,
                    header=header,
                    realization=realization,
                )
                self.a_u_map_bit_product_to_linearized_variable[k][bit_product] = (
                    self.model.addVar(vtype=GRB.BINARY, name=f"A[{i}][{k}]")
                )
                parameterized_column[i] = (
                    self.a_u_map_bit_product_to_linearized_variable[k][bit_product]
                )
            self.parameterized_columns.append(parameterized_column)
    
    def generate_linearize_alphas_constraints(
        self, k, map_bit_product_to_linearized_variable: dict[BitProduct, Var], name=""
    ) -> None:
        i = 0
        for bit_product, _ in map_bit_product_to_linearized_variable.items():
            self.add_linearized_alphas_constraints(
                self.alphas[i][k], bit_product.bit_list, name=name, ith=i, k=k
            )
            i += 1
    
    def add_linearized_alphas_constraints(
        self, alpha, bit_list: list[Bit], name="", ith=-1, k=0
    ) -> None:
        self.model.addConstr(alpha >= 0, name=f"{name}_{ith}th_more_than_zero")
        self.model.addConstr(alpha <= self.vars[k])
        sum_bits = 0
        for bit in bit_list:
            one_or_zero = 0
            if bit.sign == -1:
                one_or_zero = 1

            self.model.addConstr(
                alpha <= one_or_zero + bit.sign * bit.gurobi_var,
                name=f"{name}_{ith}th_less_than_bit_{bit.sign}",
            )
            sum_bits += one_or_zero + bit.sign * bit.gurobi_var

        n = len(bit_list)
        self.model.addConstr(
            alpha >=  -n + self.vars[k] + sum_bits,
            name=f"{name}_{ith}th_Linearized_Sum_BitProduct",
        )
    
    def generate_bit_product(
        self,
        iteration,
        node_list: list[Node],
        header: list[str],
        realization: list[int],
        consider_intervention=False,
    ):
        bit_product = BitProduct()
        logger.debug("Bit Product Generation:")
        for node in node_list:
            if consider_intervention and node.label == self.intervention.label:
                continue
            cluster_bit_gurobi_var = (
                self._get_cluster_node_bit_variable_given_parents_realization(
                    iteration, node, realization, header
                )
            )
            node_idx = header.index(node.label)
            node_value = realization[node_idx]
            binary_node_value = bin(node_value)[2:]
            node_number_of_bits = math.ceil(math.log2(node.cardinality))
            if len(binary_node_value) > node_number_of_bits:
                raise Exception(
                    f"Binary number of node {node.label} is greater than its cardinality"
                )
            else:
                binary_node_value = (
                    node_number_of_bits - len(binary_node_value)
                ) * "0" + binary_node_value
            logger.debug(
                f"Node: {node.label} Cardinality: {node.cardinality} | log2(Cardinality): {node_number_of_bits} | Binary: {binary_node_value} | Value: {node_value}"
            )

            for str_index in range(node_number_of_bits):
                sign = 1
                variable_index = node_number_of_bits - str_index - 1
                if int(binary_node_value[str_index]) == 0:
                    sign = -1
                new_bit = Bit(cluster_bit_gurobi_var[variable_index], sign)
                bit_product.add_bit(new_bit)
        return bit_product
    
    def generate_linearized_bit_products_constraints(
        self, map_bit_product_to_linearized_variable: dict[BitProduct, Var], name=""
    ) -> None:
        i = 0
        variables_sets = set()
        for bit_product, variable in map_bit_product_to_linearized_variable.items():
            self.add_linearized_bit_products_constraints(
                variable, bit_product.bit_list, name=name, ith=i, variable_set=variables_sets
            )
            i += 1
            variables_sets.add(variable.VarName)
    
    def add_linearized_bit_products_constraints(
        self, variable: Var, bit_list: list[Bit], name="", ith=-1, variable_set=set()
    ) -> None:
        if variable.VarName not in variable_set:
            self.model.addConstr(variable >= 0, name=f"{name}_{ith}th_more_than_zero")
            self.model.addConstr(variable <= 1, name=f"{name}_{ith}th_less_than_one")

        sum_bits = 0
        for bit in bit_list:
            one_or_zero = 0
            if bit.sign == -1:
                one_or_zero = 1

            self.model.addConstr(
                variable <= one_or_zero + bit.sign * bit.gurobi_var,
                name=f"{name}_{ith}th_less_than_bit_{bit.sign}",
            )
            sum_bits += one_or_zero + bit.sign * bit.gurobi_var

        n = len(bit_list)
        self.model.addConstr(
            variable >= 1 - n + sum_bits,
            name=f"{name}_{ith}th_Linearized_Sum_BitProduct",
        )
    
    def _get_cluster_node_bit_variable_given_parents_realization(
        self, iteration: int, node: Node, w_realization: list[int], w_header: list[str]
    ) -> list[Var]:
        parents_labels = [
            parent.label for parent in node.parents if not parent.is_latent
        ]
        parents_realization = [
            w_realization[w_header.index(parent_label)]
            for parent_label in parents_labels
        ]
        realization_key: str = self.get_realization_key(
            parents_labels, parents_realization
        )
        return self.cluster_bits[iteration][node.label][realization_key]

def get_node_list_realizations(node_list: list[Node]) -> list[list]:
    ranges = [range(node.cardinality) for node in node_list]
    cartesian = product(*ranges)
    matrix = [[node.label for node in node_list]]
    matrix += [list(combo) for combo in cartesian]
    return matrix

