import logging
import os
import sys
import unittest

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.disable(logging.INFO)

from pici.causal_model import CausalModel
from pici.utils._enum import DataExamplesPaths
from pici.utils.scalable_graphs_helper import (
    find_true_value_in_scalable_graphs,
    generate_scalable_string_edges,
)


class TestIsIdentifiableIntervention(unittest.TestCase):
    def setUp(self):
        self.df_xy = pd.DataFrame({"X": [0, 1] * 10, "Y": [1, 0] * 10})
        self.df_xwy = pd.DataFrame(
            {"X": [0, 1] * 10, "W": [0, 1] * 10, "Y": [1, 0] * 10}
        )
        self.df_xzy = pd.DataFrame(
            {"X": [0, 1] * 10, "Z": [0, 1] * 10, "Y": [1, 0] * 10}
        )

    def test_backdoor_non_identifiable(self):
        """
        Test the pure unobservable confounder partial identifiability case.
        """
        graph = "U -> X, U -> Y, X -> Y"
        model = CausalModel(data=self.df_xy, edges=graph, unobservables_labels=["U"])
        identifiable, method, detail = model.is_identifiable_intervention(
            interventions=[("X", 0)], target=("Y", 1)
        )
        self.assertFalse(identifiable)
        self.assertEqual(method, "unobservable_confounding")
        self.assertIsNone(detail)

    def test_frontdoor_identifiable(self):
        """
        Test the front-door identifiability case, with W as a mediator.
        """
        graph = "U1 -> X, U1 -> Y, X -> W, W -> Y, U2 -> W"
        model = CausalModel(
            data=self.df_xwy, edges=graph, unobservables_labels=["U1", "U2"]
        )
        identifiable, method, detail = model.is_identifiable_intervention(
            interventions=[("X", 0)], target=("Y", 1)
        )
        self.assertTrue(identifiable)
        self.assertEqual(method, "frontdoor")
        # The minimal observed front-door set should be {'W'}
        self.assertEqual(detail, frozenset({"W"}))

    def test_missing_intervention_raises(self):
        """
        Test the missing intervention error case.
        """
        graph = "U -> X, U -> Y, X -> Y"
        model = CausalModel(data=self.df_xy, edges=graph, unobservables_labels=["U"])
        with self.assertRaises(Exception):
            model.is_identifiable_intervention(interventions=[], target=("Y", 1))

    def test_missing_target_raises(self):
        """
        Test the missing target error case.
        """
        graph = "U -> X, U -> Y, X -> Y"
        model = CausalModel(data=self.df_xy, edges=graph, unobservables_labels=["U"])
        with self.assertRaises(Exception):
            model.is_identifiable_intervention(interventions=[("X", 0)], target=None)


class TestIdentifiableInterventionQueries(unittest.TestCase):
    def test_identifiable_queries_via_subtests(self):
        """
        Test identifiable intervention queries via subtests, using the scalable graph
        """
        cases = [
            (1, 1, DataExamplesPaths.CSV_N1M1),
            (2, 1, DataExamplesPaths.CSV_N2M1),
        ]
        unobs = ["U1", "U2", "U3"]
        interventions = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for N, M, csv_example in cases:
            with self.subTest(N=N, M=M):
                edges = generate_scalable_string_edges(N=N, M=M)
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

                        tv = find_true_value_in_scalable_graphs(
                            N, M, target_value, intervention_value, df
                        )

                        self.assertAlmostEqual(
                            identifiable_value,
                            tv,
                            places=2,
                            msg=f"Values do not match for N={N},M={M},Y={target_value},do(X={intervention_value})",
                        )


if __name__ == "__main__":
    unittest.main()
