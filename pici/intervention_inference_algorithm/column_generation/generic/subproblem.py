import logging

import gurobipy as gp
from gurobipy import GRB, Var, tupledict

from pici.graph.node import Node
from pici.intervention_inference_algorithm.column_generation.generic.bits import Bit, BitProduct, count_endogenous_parent_configurations
from pici.intervention_inference_algorithm.column_generation.generic.column_generation_orchestrator import get_node_list_realizations


logger = logging.getLogger(__name__)


import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


BIG_M = 1e4
DBG = False
MAX_ITERACTIONS_ALLOWED = 2000


class SubProblem:
    def __init__(self):
        self.model = gp.Model("subproblem")
        self.cluster_bits: dict[str, dict[str, tupledict[int, Var]]] = {}        
        self.constr = None
    
    def setup(
        self,
        reversed_ordered_considered_c_comp: list[Node],
        reversed_ordered_W_realizations: list[list],
        duals: dict[int, float],
        minimizes_objective_function: bool,
    ):
        self.model.setAttr(GRB.Attr.ModelSense, GRB.MINIMIZE)
        self.model.setParam(GRB.Param.FeasibilityTol, 1e-9)
        self.model.setParam(GRB.Param.OutputFlag, 0)
        self.model.setParam(GRB.Param.BestBdStop, 1)

        self.reversed_ordered_W_realizations = reversed_ordered_W_realizations

        self._create_cluster_bits(reversed_ordered_considered_c_comp, reversed_ordered_W_realizations)
        ###

        self.model.update()


    def _create_cluster_bits(self, reversed_ordered_considered_c_comp: list[Node], reversed_ordered_W_realizations: list[list]):
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


    def linearize(self, W_realizations: list[list], considered_c_component_in_topological_order: list[Node]) -> dict:
        # TODO: Edge cases: intervention and target, apenas desprezat na realization
        map_bit_product_to_linearized_variable: dict[BitProduct, Var] = {}
        header = W_realizations[0]
        cartesian_products = W_realizations[1:]

        for realization in cartesian_products:
            # quais são as condições para essa função?
            coef = get_coef_from_objective_function()
            bit_product = BitProduct()
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
            self.model.addConstr(variable >= 0, name=f"BOOOO")
            # TODO: Add constraints name
            self.model.addConstr(variable <= 1)

            sum_bits = 0
            for bit in bit_product.bit_list:
                one_or_zero = 0
                if bit.sign == -1:
                    one_or_zero = 1
            
                # TODO: Add constraint name
                self.model.addConstr(variable <= one_or_zero + bit.sign*bit.gurobi_var, name=f"_______")
                sum_bits += one_or_zero + bit.sign*bit.gurobi_var

            n = len(bit_product.bit_list)
            self.model.addConstr(variable >= 1 - n + sum_bits)
