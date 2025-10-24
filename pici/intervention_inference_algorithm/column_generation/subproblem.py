from itertools import product
import logging
import math

import gurobipy as gp
from gurobipy import GRB, Var, tupledict
from pandas import DataFrame

from pici.graph.node import Node
from pici.intervention_inference_algorithm.column_generation.bits import (
    Bit,
    BitProduct,
)
from pici.utils.probabilities_helper import find_conditional_probability

logger = logging.getLogger(__name__)


import os
import sys

THIS_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../.."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.path.append(os.path.abspath(os.path.join(THIS_DIR, PROJECT_ROOT)))


class SubProblem:

    def __init__(
        self,
        intervention: Node,
        target: Node,
        df: DataFrame = None,
        minimizes_objective_function=False,
    ):
        self.intervention = intervention
        self.target = target
        self.df = df
        self.model = gp.Model("subproblem")
        self.cluster_bits: dict[str, dict[str, tupledict[int, Var]]] = {}
        self.minimizes_objective_function = minimizes_objective_function

    def setup(
        self,
        reversed_ordered_considered_c_comp: list[Node],
        reversed_ordered_W_realizations: list[list],
        reversed_ordered_W: list[Node],
        symbolic_objective_function_probabilites: list[tuple],
        reversed_ordered_considered_c_comp_plus_adapted_tail: list[Node],
    ):
        self.model.setAttr(GRB.Attr.ModelSense, GRB.MINIMIZE)
        self.model.setParam(GRB.Param.FeasibilityTol, 1e-9)
        self.model.setParam(GRB.Param.OutputFlag, 0)
        self.model.setParam(GRB.Param.BestBdStop, 1)

        self.parameterized_column = {}

        self.reversed_ordered_W = reversed_ordered_W
        self.reversed_ordered_considered_c_comp = reversed_ordered_considered_c_comp
        self.reversed_ordered_W_realizations = reversed_ordered_W_realizations

        logger.debug(
            "________________________________________________________________________________"
        )
        logger.debug("FO Symbolic: ------")
        for element in symbolic_objective_function_probabilites:
            logger.debug(element)

        realization_reversed_ordered_considered_c_comp_plus_adapted_tail = (
            get_node_list_realizations(
                reversed_ordered_considered_c_comp_plus_adapted_tail
            )
        )
        self._create_cluster_bits(reversed_ordered_considered_c_comp)
        self._add_constraints_cluster_bits(reversed_ordered_considered_c_comp)
        self.model.update()

        self.objective_function_vars_not_in_W = (
            self.get_objective_function_vars_not_in_W(
                symbolic_objective_function_probabilites, reversed_ordered_W
            )
        )
        """
        Pw: são todas as probabilidades condicionais na FO em que todas as variáveis estão em W
        Pq: são todas as probabilidades condicionais na FO menos as de Pw
        """
        self.Pw, self.Pq = self.separate_objective_function_probabilities(
            symbolic_objective_function_probabilites, reversed_ordered_W
        )
        self.realization_objective_function_vars_not_in_W = get_node_list_realizations(
            self.objective_function_vars_not_in_W
        )

        self.gamma_u_map_bit_product_to_linearized_variable: dict[BitProduct, Var] = (
            self.gamma_linearize(
                reversed_ordered_considered_c_comp,
                realization_reversed_ordered_considered_c_comp_plus_adapted_tail,
                self.realization_objective_function_vars_not_in_W,
            )
        )
        self.generate_linearized_bit_products_constraints(
            self.gamma_u_map_bit_product_to_linearized_variable, name="gamma"
        )
        self.model.update()
        self.a_u_map_bit_product_to_linearized_variable: dict[BitProduct, Var] = {}

        self.get_A_u_column_and_parameterized_column(
            reversed_ordered_W_realizations, reversed_ordered_considered_c_comp
        )
        self.generate_linearized_bit_products_constraints(
            self.a_u_map_bit_product_to_linearized_variable, name="au"
        )
        self.model.update()

        logger.debug(
            "________________________________________________________________________________"
        )
        logger.debug("GammaU: ------")
        i = 0
        for (
            bit_product,
            _,
        ) in self.gamma_u_map_bit_product_to_linearized_variable.items():
            str_prod_bit = f"{bit_product.coef} * "
            for b in bit_product.bit_list:
                str_prod_bit += f"({b.sign} * {b.gurobi_var.VarName}) * "
            logger.debug(f"Element {i}th: {str_prod_bit[:len(str_prod_bit)-3]},")
            i += 1
        logger.debug(
            "________________________________________________________________________________"
        )
        print("Au: ------")
        for bit_product, _ in self.a_u_map_bit_product_to_linearized_variable.items():
            str_prod_bit = ""
            for b in bit_product.bit_list:
                str_prod_bit += f"({b.sign} * {b.gurobi_var.VarName}) * "

            print(f"{str_prod_bit[:len(str_prod_bit)-3]}, ")

    def _create_cluster_bits(self, considered_c_comp: list[Node]):
        """
        Each node in the considered c-component has a series of bits that represents each realization.
        Example:
            A = b0b1b2
            B = b0
            C = b0b1
            We've three clusters. Cluster A with 3 bits, Cluster B with one bit, and Cluster C with two bits.

        """
        j = 0
        logger.debug(f"Cluster Bits of :{considered_c_comp}")
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
            self.cluster_bits[node.label] = {}
            for i, realization in enumerate(reversed_ordered_node_parents_realizations):
                realization_key: str = self.get_realization_key(header, realization)
                logger.debug(
                    f"    {j}th - Node {node.label} Realization key: {realization_key}--{realization} ==> {node_number_of_bits} bits"
                )
                j += 1
                self.cluster_bits[node.label][realization_key] = self.model.addVars(
                    node_number_of_bits,
                    obj=0,
                    vtype=GRB.BINARY,
                    name=f"bit_realization_{i}th_of_node_{node.label}_{realization_key}",
                )

    def _add_constraints_cluster_bits(self, considered_c_comp: list[Node]):
        """
        Each node in the considered c-component has a series of bits that represents each realization.
        """
        j = 0
        logger.debug(f"Constraints fot the Cluster Bits of :{considered_c_comp}")
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
                    f"    {j}th - Constraint of Node {node.label} Realization key: {realization_key}--{realization}"
                )
                j += 1
                debug_expr_str = ""
                expr = 0
                for variable_index in range(node_number_of_bits):
                    expr += (2 ** (variable_index)) * self.cluster_bits[node.label][
                        realization_key
                    ][variable_index]
                    debug_expr_str += f"2^{variable_index} * b{variable_index} + "
                debug_expr_str += f"<= {node.cardinality-1}"
                logger.debug(debug_expr_str)
                self.model.addConstr(
                    expr <= node.cardinality - 1,
                    name=f"discrete_constraint_of_node_{node.label}_{realization_key}",
                )

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
        Pw todo mundo está em W
        Pq: P - Pw
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

    def get_realization_key(self, header: list[str], realization: list[int]) -> str:
        if len(header) != len(realization):
            raise ValueError(
                "Lists with different sizes. Header and Realization should have the same size."
            )
        realization_key = ""
        for i in range(len(header)):
            realization_key += f"{header[i]}={realization[i]},"
        return realization_key[: len(realization_key) - 1]

    def update(self, duals):
        """
        Change the objective functions coefficients.
        """
        self.model.setAttr(
            "obj", self.parameterized_column, [-duals[dualKey] for dualKey in duals]
        )
        self.model.update()

    def gamma_linearize(
        self,
        reversed_ordered_considered_c_comp: list[Node],
        realization_reversed_ordered_considered_c_comp_plus_adapted_tail: list,
        realization_objective_function_vars_not_in_W: list,
    ) -> dict:
        """
        Gera o Yu (Gamma U): gamma_u_map_bit_product_to_linearized_variable
        Mapeia o produtório de bits e seu coef a uma linearização
        """
        gamma_u_map_bit_product_to_linearized_variable: dict[BitProduct, Var] = {}
        header = realization_reversed_ordered_considered_c_comp_plus_adapted_tail[0]
        cartesian_products = (
            realization_reversed_ordered_considered_c_comp_plus_adapted_tail[1:]
        )

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
            if not self.minimizes_objective_function:
                coef = -1 * coef
            bit_product: BitProduct = self.generate_bit_product(
                node_list=reversed_ordered_considered_c_comp,
                header=header,
                realization=realization,
                consider_intervention=True,
            )
            bit_product.set_coef(coef)
            gamma_u_map_bit_product_to_linearized_variable[bit_product] = (
                self.model.addVar(
                    obj=coef, vtype=GRB.BINARY, name="linearization_auxiliary_variable"
                )
            )
        return gamma_u_map_bit_product_to_linearized_variable

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
        # TODO: rename this var
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
            coef_parcial = 1
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
                coef_parcial *= curr
                logger.debug(
                    f"P({q_target.label}={q_target.value}|{str_q[:len(str_q)-2]}) == {curr}"
                )
            logger.debug(f"Coef_Parcial: {coef_parcial}")
            coefq += coef_parcial
            logger.debug(f"coefW: {coefw}")
            logger.debug(f"coefq: {coefq}")
            logger.debug(f"coef: {coefw*coefq}")
            logger.debug("----")
        return coefq * coefw

    def _get_cluster_node_bit_variable_given_parents_realization(
        self, node: Node, w_realization: list[int], w_header: list[str]
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
        return self.cluster_bits[node.label][realization_key]

    def get_A_u_column_and_parameterized_column(
        self, total_w_realization, considered_c_component_in_topological_order
    ):
        header = total_w_realization[0]
        total_w_realization = total_w_realization[1:]
        for i, realization in enumerate(total_w_realization):
            bit_product: BitProduct = self.generate_bit_product(
                node_list=considered_c_component_in_topological_order,
                header=header,
                realization=realization,
            )
            self.a_u_map_bit_product_to_linearized_variable[bit_product] = (
                self.model.addVar(vtype=GRB.BINARY)
            )
            self.parameterized_column[i] = (
                self.a_u_map_bit_product_to_linearized_variable[bit_product]
            )

    def generate_bit_product(
        self,
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
                    node, realization, header
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
                f"Node: {node.label} Card: {node.cardinality} | K: {node_number_of_bits} | Binario: {binary_node_value} | Valor: {node_value}"
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
        for bit_product, variable in map_bit_product_to_linearized_variable.items():
            self.add_linearized_bit_products_constraints(
                variable, bit_product.bit_list, name=name, ith=i
            )
            i += 1

    def add_linearized_bit_products_constraints(
        self, variable: Var, bit_list: list[Bit], name="", ith=-1
    ) -> None:
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


def get_node_list_realizations(node_list: list[Node]) -> list[list]:
    ranges = [range(node.cardinality) for node in node_list]
    cartesian = product(*ranges)
    matrix = [[node.label for node in node_list]]
    matrix += [list(combo) for combo in cartesian]
    return matrix
