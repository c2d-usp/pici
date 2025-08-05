import os
import sys
import unittest

import pandas as pd
import numpy as np
import logging

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.disable(logging.INFO)

from causal_reasoning.utils._enum import DataExamplesPaths
from auxiliary import genGraph, true_value
from causal_reasoning.causal_model import CausalModel


class TestIsIdentifiableIntervention(unittest.TestCase):
    def setUp(self):
        self.df_xy = pd.DataFrame({
            'X': [0, 1] * 10,
            'Y': [1, 0] * 10
        })
        self.df_xwy = pd.DataFrame({
            'X': [0, 1] * 10,
            'W': [0, 1] * 10,
            'Y': [1, 0] * 10
        })
        self.df_xzy = pd.DataFrame({
            'X': [0, 1] * 10,
            'Z': [0, 1] * 10,
            'Y': [1, 0] * 10
        })

    def test_backdoor_non_identifiable(self):
        # Pure confounder U -> X, U -> Y
        graph = "U -> X, U -> Y, X -> Y"
        model = CausalModel(
            data=self.df_xy,
            edges=graph,
            unobservables_labels=['U']
        )
        identifiable, method, detail = model.is_identifiable_intervention(
            interventions=[('X', 0)],
            target=('Y', 1)
        )
        self.assertFalse(identifiable)
        self.assertEqual(method, "unobservable_confounding")
        self.assertIsNone(detail)

    def test_frontdoor_identifiable(self):
        # Front-door: U1 -> X,Y; X -> W -> Y; U2 -> W (harmless latent on W)
        graph = "U1 -> X, U1 -> Y, X -> W, W -> Y, U2 -> W"
        model = CausalModel(
            data=self.df_xwy,
            edges=graph,
            unobservables_labels=['U1', 'U2']
        )
        identifiable, method, detail = model.is_identifiable_intervention(
            interventions=[('X', 0)],
            target=('Y', 1)
        )
        self.assertTrue(identifiable)
        self.assertEqual(method, "frontdoor")
        # The minimal observed front-door set should be {'W'}
        self.assertEqual(detail, frozenset({'W'}))

    def test_iv_identifiable(self):
        # IV: U1 -> X,Y; U2 -> Z; Z -> X -> Y
        graph = "U1 -> X, U1 -> Y, U2 -> Z, Z -> X, X -> Y"
        model = CausalModel(
            data=self.df_xzy,
            edges=graph,
            unobservables_labels=['U1', 'U2']
        )
        identifiable, method, detail = model.is_identifiable_intervention(
            interventions=[('X', 0)],
            target=('Y', 1)
        )
        self.assertTrue(identifiable)
        self.assertEqual(method, "instrumental_variable")
        # The identified instrument should be 'Z'
        self.assertEqual(detail, 'Z')

    def test_missing_intervention_raises(self):
        graph = "U -> X, U -> Y, X -> Y"
        model = CausalModel(
            data=self.df_xy,
            edges=graph,
            unobservables_labels=['U']
        )
        with self.assertRaises(Exception):
            model.is_identifiable_intervention(
                interventions=[],
                target=('Y', 1)
            )

    def test_missing_target_raises(self):
        graph = "U -> X, U -> Y, X -> Y"
        model = CausalModel(
            data=self.df_xy,
            edges=graph,
            unobservables_labels=['U']
        )
        with self.assertRaises(Exception):
            model.is_identifiable_intervention(
                interventions=[('X', 0)],
                target=None
            )

class TestIdentifiableInterventionQueries(unittest.TestCase):
    def test_identifiable_queries_via_subtests(self):
        cases = [
            (1, 1, DataExamplesPaths.CSV_N1M1),
            (2, 1, DataExamplesPaths.CSV_N2M1),
        ]
        unobs = ["U1", "U2", "U3"]
        interventions = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for N, M, csv_example in cases:
            with self.subTest(N=N, M=M):
                edges = genGraph(N=N, M=M)
                df = pd.read_csv(os.path.join(PROJECT_ROOT, csv_example.value))
                df["U3"] = np.random.binomial(1, 0.5, size=len(df))

                model = CausalModel(
                    data=df,
                    edges=edges,
                    unobservables_labels=unobs,
                )

                for target_value, intervention_value in interventions:
                    with self.subTest(
                        N=N, M=M, target=target_value, intervention=intervention_value
                    ):
                        model.set_interventions([("X", intervention_value)])
                        model.set_target(("Y", target_value))

                        identifiable_value = model.identifiable_intervention_query()

                        tv = true_value(N, M, target_value, intervention_value, df)

                        self.assertAlmostEqual(
                            identifiable_value,
                            tv,
                            places=2,
                            msg=f"Values do not match for N={N},M={M},Y={target_value},do(X={intervention_value})",
                        )

if __name__ == "__main__":
    unittest.main()
