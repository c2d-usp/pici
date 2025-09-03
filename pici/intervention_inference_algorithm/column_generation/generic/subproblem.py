import logging

import gurobipy as gp
from gurobipy import GRB, Var, tupledict

from pici.graph.node import Node
from pici.intervention_inference_algorithm.column_generation.generic.bits import Bit, BitProduct, count_endogenous_parent_configurations


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
        self.cluster_bits: dict[str, tupledict[int, Var]] = {}        
        self.constr = None
    
    def setup(
        self,
        considered_c_comp_reversed_ordered: list[Node],
        W_realizations: list[list],
        duals: dict[int, float],
        minimizes_objective_function: bool,
    ):
        self.model.setAttr(GRB.Attr.ModelSense, GRB.MINIMIZE)
        self.model.setParam(GRB.Param.FeasibilityTol, 1e-9)
        self.model.setParam(GRB.Param.OutputFlag, 0)
        self.model.setParam(GRB.Param.BestBdStop, 1)

        self._create_cluster_bits(considered_c_comp_reversed_ordered)
        self.W_realizations = W_realizations
        ###

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
        for node in considered_c_comp:
            node_cluster_size = count_endogenous_parent_configurations(node)
            variable_bits_name: list[str] = []
            for i in range(node_cluster_size):
                variable_bits_name.append(f"cluster_of_node_{node.label}_bit_{i}")

            self.cluster_bits[node.label] = self.model.addVars(
                node_cluster_size, obj=0, vtype=GRB.BINARY, name=variable_bits_name
            )

    def update(self, duals):
        """
        Change the objective functions coefficients.
        """
        self.model.setAttr(
            "obj", self.bitsParametric, [-duals[dualKey] for dualKey in duals]
        )
        self.model.update()


    def linearize(self, W_realizations: list[list], considered_c_component_in_topological_order: list[Node]) -> dict:
        map_bit_product_to_linearized_variable: dict[BitProduct, Var] = {}
        header = W_realizations[0]
        cartesian_products = W_realizations[1:]

        for realization in cartesian_products:
            coef = get_coef_from_objective_function()
            bit_product = BitProduct()
            # Aqui eu não vou passar várias vezes pelo mesmo "conjunto"
            # W         = A,B,C,D,F
            # CONS_COMP = A,B,C
            # LOGO, Vou passar por A=0,B=0,C=0 4 vezes... é isso?
            for node in considered_c_component_in_topological_order:
                # Já está! Ordenar Pa(node) para ver a ordem dos bits na cartesian_products
                considered_c_component_realization = self._get_considered_c_component_realization(header, realization, considered_c_component_in_topological_order)
                # O bit depende do valor do current node? Não seria só dos pais?
                bit_gurobi_var = self._get_bit_var_given_parents_rlz_and_current_node(considered_c_component_in_topological_order, considered_c_component_realization)

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

    def _get_considered_c_component_realization(self, header: list, realization: list, considered_c_component: list[Node]) -> list[int]:
        indices = [header.index(node.label) for node in considered_c_component]
        return [realization[i] for i in indices]


    def _get_bit_var_given_parents_rlz_and_current_node(self, current_node: Node, considered_c_component: list[Node], considered_c_component_realization: list[int]):
        '''
        A=0,B=0,C=0 --> cluster bits of current node --> apontador para objeto gurobi
        A=0,B=0,C=1
        '''

        node_bit = self.cluster_bits[current_node.label][considered_c_component_realization]


        pass

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
