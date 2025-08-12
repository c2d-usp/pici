import logging
import os
import sys
import unittest

import networkx as nx
import pandas as pd

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.disable(logging.INFO)

from pici.causal_model import CausalModel
from pici.utils._enum import DataExamplesPaths


class TestInferenceAlgorithm(unittest.TestCase):
    def test_binary_balke_pearl(self):
        edges = "Z -> X, X -> Y, U1 -> X, U1 -> Y, U2 -> Z"
        card = {"Z": 2, "X": 2, "Y": 2, "U1": 0, "U2": 0}
        unobs = ["U1", "U2"]
        df = pd.read_csv(
            os.path.join(PROJECT_ROOT, DataExamplesPaths.CSV_BALKE_PEARL_EXAMPLE.value)
        )

        model = CausalModel(
            data=df, edges=edges, custom_cardinalities=card, unobservables_labels=unobs
        )
        model.set_interventions([("X", 1)])
        model.set_target(("Y", 1))

        self.assertFalse(
            model.are_d_separated_in_complete_graph(["Z"], ["Y"], ["X"]),
            msg="Balke-Pearl: complete graph should not d-separate Z and Y given X",
        )
        self.assertTrue(
            model.are_d_separated_in_intervened_graph(["Z"], ["Y"], ["X"]),
            msg="Balke-Pearl: intervened graph should d-separate Z and Y given X",
        )

        lower, upper = model.intervention_query()
        expected_lower, expected_upper = (0.45000000000000007, 0.5199999999999999)
        self.assertAlmostEqual(
            float(lower),
            expected_lower,
            places=3,
            msg=f"Balke-Pearl lower bound mismatch: expected {expected_lower}, got {lower}",
        )
        self.assertAlmostEqual(
            float(upper),
            expected_upper,
            places=3,
            msg=f"Balke-Pearl upper bound mismatch: expected {expected_upper}, got {upper}",
        )

    def test_discrete_iv_random(self):
        edges = "Z -> X, X -> Y, U1 -> X, U1 -> Y, U2 -> Z"
        card = {"Z": 4, "X": 3, "Y": 2, "U1": 0, "U2": 0}
        unobs = ["U1", "U2"]
        df = pd.read_csv(
            os.path.join(
                PROJECT_ROOT, DataExamplesPaths.CSV_DISCRETE_IV_RANDOM_EXAMPLE.value
            )
        )

        model = CausalModel(
            data=df, edges=edges, custom_cardinalities=card, unobservables_labels=unobs
        )
        model.set_interventions([("X", 2)])
        model.set_target(("Y", 1))

        lower, upper = model.intervention_query()
        expected_lower, expected_upper = (0.17221135029354206, 0.8160779537149818)
        self.assertAlmostEqual(
            float(lower),
            expected_lower,
            places=3,
            msg=f"Discrete IV lower bound mismatch: expected {expected_lower}, got {lower}",
        )
        self.assertAlmostEqual(
            float(upper),
            expected_upper,
            places=3,
            msg=f"Discrete IV upper bound mismatch: expected {expected_upper}, got {upper}",
        )

    def test_binary_copilot_example(self):
        edges = (
            "X -> Y, X -> D, D -> Y, E -> D, U1 -> Y, U1 -> X,"
            " U2 -> D, U3 -> E, U1 -> F"
        )
        card = {
            "X": 2,
            "Y": 2,
            "D": 2,
            "E": 2,
            "F": 2,
            "U1": 0,
            "U2": 0,
            "U3": 0,
        }
        unobs = ["U1", "U2", "U3"]
        df = pd.read_csv(
            os.path.join(PROJECT_ROOT, DataExamplesPaths.CSV_COPILOT_EXAMPLE.value)
        )

        model = CausalModel(
            data=df, edges=edges, custom_cardinalities=card, unobservables_labels=unobs
        )
        model.set_interventions([("X", 1)])
        model.set_target(("Y", 1))

        self.assertFalse(
            model.are_d_separated_in_complete_graph(["E"], ["X"], ["D"]),
            msg="Copilot: complete graph should not d-separate E and X given D",
        )
        lower, upper = model.intervention_query()
        expected_lower, expected_upper = (0.3570355041940286, 0.8560355041940286)
        self.assertAlmostEqual(
            float(lower),
            expected_lower,
            places=3,
            msg=f"Copilot lower bound mismatch: expected {expected_lower}, got {lower}",
        )
        self.assertAlmostEqual(
            float(upper),
            expected_upper,
            places=3,
            msg=f"Copilot upper bound mismatch: expected {expected_upper}, got {upper}",
        )

        model.set_interventions([("D", 1)])
        self.assertTrue(
            model.are_d_separated_in_intervened_graph(["E"], ["Y"], ["D"]),
            msg="Copilot: intervened graph should d-separate E and Y given D",
        )

    # TODO: remove double intervention
    @unittest.skip("double intervention has a bug")
    def test_double_intervention_binary_balke_pearl(self):
        edges = "Z -> X, X -> Y, U1 -> X, U1 -> Y, U2 -> Z"
        cardinalities = {"Z": 2, "X": 2, "Y": 2, "U1": 0, "U2": 0}
        unobs = ["U1", "U2"]
        target = "Y"
        target_value = 1
        rel_path = DataExamplesPaths.CSV_BALKE_PEARL_EXAMPLE.value
        csv_path = os.path.join(PROJECT_ROOT, rel_path)
        df = pd.read_csv(csv_path)

        model = CausalModel(
            data=df,
            edges=edges,
            custom_cardinalities=cardinalities,
            unobservables_labels=unobs,
            interventions=[("X", 1), ("Z", 1)],
            target=(target, target_value),
        )

        self.assertFalse(model.are_d_separated_in_complete_graph(["Z"], ["Y"], ["X"]))
        expected_lower, expected_upper = (0.07680001854838515, 0.09309998464330864)
        expected_lower = round(expected_lower, 3)
        expected_upper = round(expected_upper, 3)
        lower, upper = model.intervention_query()
        lower = round(float(lower), 3)
        upper = round(float(upper), 3)
        self.assertEqual(lower, expected_lower)
        self.assertEqual(upper, expected_upper)

    def test_simple_counfoundness(self):
        edges = "U1 -> X, U1 -> Y"
        unobs = ["U1"]
        df = pd.read_csv(
            os.path.join(PROJECT_ROOT, DataExamplesPaths.CSV_BALKE_PEARL_EXAMPLE.value)
        )

        model = CausalModel(
            data=df,
            edges=edges,
            unobservables_labels=unobs,
        )

        self.assertTrue(
            model.are_d_separated_in_complete_graph(["X"], ["Y"], ["U1"]),
            msg="Simple confounding: X and Y should be d-separated given U1",
        )

    def test_incident_scenario(self):
        edges_list = [
            ("DB_Change", "DB_Latency"),
            ("DB_Latency", "MS-B_Latency"),
            ("MS-B_Latency", "MS-B_Error"),
            ("MS-B_Latency", "MS-A_Latency"),
            ("MS-B_Error", "MS-A_Error"),
            ("MS-A_Latency", "MS-A_Threads"),
            ("MS-A_Threads", "MS-A_Crash"),
            ("MS-A_Error", "Outage"),
            ("MS-A_Crash", "Outage"),
            ("HeavyTraffic", "MS-B_Latency"),
            ("HeavyTraffic", "MS-A_Latency"),
            ("Unob_helper_1", "DB_Change"),
            ("Unob_helper_2", "DB_Latency"),
            ("Unob_helper_3", "MS-B_Error"),
            ("Unob_helper_4", "MS-A_Error"),
            ("Unob_helper_5", "MS-A_Threads"),
            ("Unob_helper_6", "MS-A_Crash"),
            ("Unob_helper_7", "Outage"),
        ]
        latent = [
            n
            for n in nx.topological_sort(nx.DiGraph(edges_list))
            if n.startswith("Unob_") or n == "HeavyTraffic"
        ]
        df = pd.read_csv(
            os.path.join(
                PROJECT_ROOT, DataExamplesPaths.NEW_MEDIUM_SCALE_OUTAGE_INCIDENT.value
            ),
            index_col=0,
        )
        model = CausalModel(
            data=df, edges=nx.DiGraph(edges_list), unobservables_labels=latent
        )
        self.assertTrue(
            model.are_d_separated_in_complete_graph(
                ["MS-A_Latency"], ["DB_Change"], ["DB_Latency"]
            ),
            msg="Incident scenario: MS-A_Latency and DB_Change should be d-separated given DB_Latency",
        )


if __name__ == "__main__":
    unittest.main()
