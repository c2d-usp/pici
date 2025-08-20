import logging
import os
import sys
import unittest

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.disable(logging.INFO)

from pici.graph.node import Node
from pici.intervention_inference_algorithm.column_generation.generic.bits import (
    generate_optimization_problem_bit_list,
)


class TestBitsCalculation(unittest.TestCase):
    def test_iv_binary_correct(self):
        """
        Case 'Z -> X, X -> Y, U1 -> X, U1 -> Y'
        Expected: 4
        """
        node_label = "Z"
        node_cardinality = 2
        Z = Node(
            children=[],
            parents=[],
            latent_parent=None,
            is_latent=False,
            label=node_label,
            cardinality=node_cardinality,
        )
        node_label = "X"
        node_cardinality = 2
        X = Node(
            children=[],
            parents=[Z],
            latent_parent=None,
            is_latent=False,
            label=node_label,
            cardinality=node_cardinality,
        )
        node_label = "Y"
        node_cardinality = 2
        Y = Node(
            children=[],
            parents=[X],
            latent_parent=None,
            is_latent=False,
            label=node_label,
            cardinality=node_cardinality,
        )
        node_label = "U1"
        node_cardinality = 0
        U1 = Node(
            children=[Y, X],
            parents=[],
            latent_parent=None,
            is_latent=True,
            label=node_label,
            cardinality=node_cardinality,
        )

        Z.children = [X]
        X.parents = [U1, Z]
        X.children = [Y]
        X.latent_parent = U1
        Y.latent_parent = U1
        Y.parents = [X]

        bits_list = generate_optimization_problem_bit_list(X)
        self.assertEqual(
            int(len(bits_list)),
            4,
            msg=f"Values do not match for case 'Z -> X, X -> Y, U1 -> X, U1 -> Y'",
        )

    def test_intervention_without_endogenous_parent_correct(self):
        """
        Case 'Z -> K, W -> K, X -> Y, U1 -> K, U1 -> X'
        Expected: 7
        """
        node_label = "Z"
        node_cardinality = 2
        Z = Node(
            children=[],
            parents=[],
            latent_parent=None,
            is_latent=False,
            label=node_label,
            cardinality=node_cardinality,
        )
        node_label = "W"
        node_cardinality = 3
        W = Node(
            children=[],
            parents=[],
            latent_parent=None,
            is_latent=False,
            label=node_label,
            cardinality=node_cardinality,
        )
        node_label = "K"
        node_cardinality = 25
        K = Node(
            children=[],
            parents=[],
            latent_parent=None,
            is_latent=False,
            label=node_label,
            cardinality=node_cardinality,
        )

        node_label = "X"
        node_cardinality = 90
        X = Node(
            children=[],
            parents=[],
            latent_parent=None,
            is_latent=False,
            label=node_label,
            cardinality=node_cardinality,
        )
        node_label = "Y"
        node_cardinality = 90
        Y = Node(
            children=[],
            parents=[X],
            latent_parent=None,
            is_latent=False,
            label=node_label,
            cardinality=node_cardinality,
        )
        node_label = "U1"
        node_cardinality = 90
        U1 = Node(
            children=[K, X],
            parents=[],
            latent_parent=None,
            is_latent=True,
            label=node_label,
            cardinality=node_cardinality,
        )

        K.parents = [Z, W, U1]
        Z.children = [K]
        W.children = [K]
        X.parents = [U1]
        X.children = [Y]
        X.latent_parent = U1
        Y.parents = [X]

        bits_list = generate_optimization_problem_bit_list(X)
        self.assertEqual(
            int(len(bits_list)),
            7,
            msg=f"Values do not match for case 'Z -> K, W -> K, X -> Y, U1 -> K, U1 -> X'",
        )

    def test_error_child_without_latentParent(self):
        """
        Case 'Z -> X, X -> Y, U1 -> X, U1 -> Y'
        Expected: ERROR
        """
        node_label = "Z"
        node_cardinality = 2
        Z = Node(
            children=[],
            parents=[],
            latent_parent=None,
            is_latent=False,
            label=node_label,
            cardinality=node_cardinality,
        )
        node_label = "X"
        node_cardinality = 2
        X = Node(
            children=[],
            parents=[],
            latent_parent=None,
            is_latent=False,
            label=node_label,
            cardinality=node_cardinality,
        )
        node_label = "Y"
        node_cardinality = 2
        Y = Node(
            children=[],
            parents=[],
            latent_parent=None,
            is_latent=False,
            label=node_label,
            cardinality=node_cardinality,
        )
        node_label = "U1"
        node_cardinality = 0
        U1 = Node(
            children=[],
            parents=[],
            latent_parent=None,
            is_latent=True,
            label=node_label,
            cardinality=node_cardinality,
        )

        Z.children = [X]
        X.parents = [U1, Z]
        X.children = [Y]
        # Error cause: # X.latentParent = U1
        Y.latent_parent = U1
        Y.parents = [X]
        U1.children = [X, Y]
        with self.assertRaisesRegex(
            AttributeError,
            "'NoneType' object has no attribute 'children'",
            msg=f"Expects Exception raised.",
        ):
            generate_optimization_problem_bit_list(X)


if __name__ == "__main__":
    unittest.main()
