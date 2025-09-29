import logging

import gurobipy as gp
from gurobipy import GRB, Var, tupledict
from pandas import DataFrame

from pici.graph.node import Node
from pici.intervention_inference_algorithm.column_generation.generic.bits import Bit, BitProduct, count_endogenous_parent_configurations
from pici.intervention_inference_algorithm.column_generation.generic.column_generation_orchestrator import get_node_list_realizations
from pici.utils.probabilities_helper import find_conditional_probability


logger = logging.getLogger(__name__)


import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


BIG_M = 1e4
DBG = False
MAX_ITERACTIONS_ALLOWED = 2000


class SubProblem:
    def __init__(self, df: DataFrame = None):
        self.df = df
        self.model = gp.Model("subproblem")
        self.cluster_bits: dict[str, dict[str, tupledict[int, Var]]] = {}        
        self.constr = None
    
    def setup(
        self,
        reversed_ordered_considered_c_comp: list[Node],
        reversed_ordered_W_realizations: list[list],
        reversed_ordered_W: list[Node],
        symbolic_objective_function_probabilites: list[tuple],
        duals: dict[int, float],
        minimizes_objective_function: bool,
    ):
        self.model.setAttr(GRB.Attr.ModelSense, GRB.MINIMIZE)
        self.model.setParam(GRB.Param.FeasibilityTol, 1e-9)
        self.model.setParam(GRB.Param.OutputFlag, 0)
        self.model.setParam(GRB.Param.BestBdStop, 1)

        self.reversed_ordered_W = reversed_ordered_W

        self.reversed_ordered_W_realizations = reversed_ordered_W_realizations

        self._create_cluster_bits(reversed_ordered_considered_c_comp)
        self.objective_function_vars_not_in_W = self.get_objective_function_vars_not_in_W(symbolic_objective_function_probabilites, reversed_ordered_W)
        self.Pw, self.Pq = self.separate_objective_function_probabilities(symbolic_objective_function_probabilites, reversed_ordered_W)

        self.realization_objective_function_vars_not_in_W = get_node_list_realizations(self.objective_function_vars_not_in_W)

        self.model.update()

    def get_objective_function_vars_not_in_W(self, symbolic_objective_function_probabilites, W) -> list[Node]:
        objective_function_vars_not_in_W = set()
        for conditional_probability in symbolic_objective_function_probabilites:
            probability_target, conditioned_nodes = conditional_probability
            if probability_target not in W:
                objective_function_vars_not_in_W.add(probability_target)
            for node in conditioned_nodes:
                if node not in W:
                    objective_function_vars_not_in_W.add(node)
        return [node for node in objective_function_vars_not_in_W]
    
    def separate_objective_function_probabilities(self, symbolic_objective_function_probabilites, W) -> tuple[list[tuple], list[tuple]]:
        probabilities_of_objective_function_vars_not_in_W = []
        P_W = []
        for conditional_probability in symbolic_objective_function_probabilites:
            probability_target, conditioned_nodes = conditional_probability
            if probability_target not in W:
                P_W.append(conditional_probability)
                continue
            if any(node in W for node in conditioned_nodes):
                P_W.append(conditional_probability)
                continue
            probabilities_of_objective_function_vars_not_in_W.append(conditional_probability)


        return (P_W, probabilities_of_objective_function_vars_not_in_W)

    def _create_cluster_bits(self, reversed_ordered_considered_c_comp: list[Node]):
        """
        Each node in the considered c-component has a series of bits that represents each realization.
        Example:
            A = b0b1b2
            B = b0
            C = b0b1
            We've three clusters. Cluster A with 3 bits, Cluster B with one bit, and Cluster C with two bits.

        """
        for node in reversed_ordered_considered_c_comp:
            reversed_ordered_node_parents_realizations: list[list] = get_node_list_realizations(node.parents)
            header = reversed_ordered_node_parents_realizations[0]
            reversed_ordered_node_parents_realizations = reversed_ordered_node_parents_realizations[1:]
            
            self.cluster_bits[node.label] = {}
            for i, realization in enumerate(reversed_ordered_node_parents_realizations):
                realization_key: str = self.get_realization_key(header, realization)
                self.cluster_bits[node.label][realization_key] = self.model.addVar(
                    obj=0, vtype=GRB.BINARY, name=f"bit_realization_{i}_of_node_{node.label}"
                )

    def get_realization_key(self, header: list[str], realization: list[int]) -> str:
        if len(header) != len(realization):
            raise ValueError('Lists with different sizes. Header and Realization should have the same size.')
        realization_key = ""
        for i in range(len(header)):
            realization_key += f"{header[i]}={realization[i]},"
        return realization_key[:len(realization_key)-1]

    def update(self, duals):
        """
        Change the objective functions coefficients.
        """
        self.model.setAttr(
            "obj", self.bitsParametric, [-duals[dualKey] for dualKey in duals]
        )
        self.model.update()



    def get_coef_from_objective_function(self, w_header: list, w_realization: list):
        coefw = 1
        for w_conditional_probability in self.Pw:
            w_target, w_conditioned = w_conditional_probability

            w_target.value = w_realization[w_header.index(w_target.label)]
            for node in w_conditioned:
                node.value = w_realization[w_header.index(node.label)]

            coefw *= find_conditional_probability(dataFrame=self.df, target_realization=[w_target], condition_realization=w_conditioned)
            

        if len(self.objective_function_vars_not_in_W) <= 0:
            return coefw

        coefq = 0
        q_header = self.realization_objective_function_vars_not_in_W[0]
        q_realizations = self.realization_objective_function_vars_not_in_W[1:]

        for q_realization in q_realizations:
            coef_parcial = 1
            for q_conditional_probability in self.Pq:
                q_target, q_conditioned = q_conditional_probability
                
                if q_target in self.reversed_ordered_W:
                    q_target.value = w_realization[w_header.index(q_target.label)]
                else:
                    q_target.value = q_realization[q_header.index(q_target.label)]
                
                for node in q_conditioned:
                    if node in self.reversed_ordered_W:
                        node.value = w_realization[w_header.index(node.label)]
                    else:
                        node.value = q_realization[q_header.index(node.label)]
                coef_parcial *= find_conditional_probability(dataFrame=self.df, target_realization=[q_target],condition_realization=q_conditioned)
            coefq += coef_parcial
        return coefq * coefw


    def linearize(self, W_realizations: list[list], considered_c_component_in_topological_order: list[Node]) -> dict:
        # TODO: Edge cases: intervention and target, apenas desprezat na realization
        map_bit_product_to_linearized_variable: dict[BitProduct, Var] = {}
        header = W_realizations[0]
        cartesian_products = W_realizations[1:]

        for realization in cartesian_products:
            # quais são as condições para essa função?
            coef = self.get_coef_from_objective_function(header, realization)
            bit_product = BitProduct()
            # TODO: Devo desprezar a intervention e a target aqui?
            for node in considered_c_component_in_topological_order:
                
                parents_label = [parent.label for parent in node.parents]
                parents_realization = [realization[header.index(parent_label)] for parent_label in parents_label]
                realization_key: str = self.get_realization_key(parents_label, parents_realization)
                bit_gurobi_var = self.cluster_bits[node.label][realization_key]

                node_idx = header.index(node.label)
                sign = 1
                if realization[node_idx] == 0:
                    sign = -1
                new_bit = Bit(bit_gurobi_var, sign)

                bit_product.add_bit(new_bit)
            
            # TODO: Add variable name
            map_bit_product_to_linearized_variable[bit_product] = self.model.addVar(obj=coef, vtype=GRB.BINARY)
        
        '''
        TODO: Pode ser que o gurobi sabe linearizar o produtório.
        Basicamente teriamos uma lista de produtórios ao inveés de um dicionário mapeando uma nova variável.
        Para cada produtório:
            addConstr(0 <= produtorio <= 1)
        '''

        return map_bit_product_to_linearized_variable

    def _get_node_bit_variable_given_parents_realization(self, node: Node, w_realization: list[int], w_header: list[str]) -> Var:
        parents_label = [parent.label for parent in node.parents]
        parents_realization = [w_realization[w_header.index(parent_label)] for parent_label in parents_label]
        realization_key: str = self.get_realization_key(parents_label, parents_realization)
        return self.cluster_bits[node.label][realization_key]


    def add_linearized_bit_products_constraints(self, map_bit_product_to_linearized_variable: dict[BitProduct, Var]) -> None:
        for bit_product, variable in map_bit_product_to_linearized_variable.items():
            # TODO: Add constraint name
            self.model.addConstr(variable >= 0)
            # TODO: Add constraints name
            self.model.addConstr(variable <= 1)

            sum_bits = 0
            for bit in bit_product.bit_list:
                one_or_zero = 0
                if bit.sign == -1:
                    one_or_zero = 1
            
                # TODO: Add constraint name
                self.model.addConstr(variable <= one_or_zero + bit.sign*bit.gurobi_var)
                sum_bits += one_or_zero + bit.sign*bit.gurobi_var

            n = len(bit_product.bit_list)
            self.model.addConstr(variable >= 1 - n + sum_bits)
