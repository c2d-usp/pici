import logging

import gurobipy as gp
from gurobipy import GRB, Var, tupledict

from pici.graph.node import Node
from pici.intervention_inference_algorithm.column_generation.generic.bits import count_endogenous_parent_configurations


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
        self.cluster_bits: list[tupledict[int, Var]] = []        
        self.constr = None
    
    def setup(
        self,
        considered_c_comp_reversed_ordered: list[Node],
        duals: dict[int, float],
        minimizes_objective_function: bool,
    ):
        self.model.setAttr(GRB.Attr.ModelSense, GRB.MINIMIZE)
        self.model.setParam(GRB.Param.FeasibilityTol, 1e-9)
        self.model.setParam(GRB.Param.OutputFlag, 0)
        self.model.setParam(GRB.Param.BestBdStop, 1)

        self._create_cluster_bits(considered_c_comp_reversed_ordered)
        
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
        for cluster_index, node in enumerate(   ):
            node_cluster_size = count_endogenous_parent_configurations(node)
            variable_bits_name: list[str] = []
            for i in range(node_cluster_size):
                variable_bits_name.append(f"cluster_{cluster_index}_bit_{i}_node_{node.label}")

            self.cluster_bits[cluster_index] = self.model.addVars(
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
