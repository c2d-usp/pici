import os
import sys
import unittest

import networkx as nx
import pandas as pd
import numpy as np
import logging

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.disable(logging.INFO)

from causal_reasoning.causal_model import CausalModel
from causal_reasoning.utils._enum import DataExamplesPaths


class TestInferenceAlgorithm(unittest.TestCase):
    def test_binary_balke_pearl_example(self):
        balke_input = "Z -> X, X -> Y, U1 -> X, U1 -> Y, U2 -> Z"
        balke_cardinalities = {"Z": 2, "X": 2, "Y": 2, "U1": 0, "U2": 0}
        balke_unobs = ["U1", "U2"]
        balke_target = "Y"
        balke_target_value = 1
        balke_intervention = "X"
        balke_intervention_value = 1
        rel_path = DataExamplesPaths.CSV_BALKE_PEARL_EXAMPLE.value
        balke_csv_path = os.path.join(PROJECT_ROOT, rel_path)
        balke_df = pd.read_csv(balke_csv_path)

        model = CausalModel(
            data=balke_df,
            edges=balke_input,
            custom_cardinalities=balke_cardinalities,
            unobservables_labels=balke_unobs,
        )

        model.set_interventions([(balke_intervention, balke_intervention_value)])
        model.set_target((balke_target, balke_target_value))

        self.assertFalse(model.are_d_separated_in_complete_graph(["Z"], ["Y"], ["X"]))
        expected_lower, expected_upper = (0.45000000000000007, 0.5199999999999999)
        expected_lower = round(expected_lower, 3)
        expected_upper = round(expected_upper, 3)
        lower, upper = model.inference_intervention_query()
        lower = round(float(lower), 3)
        upper = round(float(upper), 3)
        self.assertEqual(lower, expected_lower)
        self.assertEqual(upper, expected_upper)
        self.assertTrue(model.are_d_separated_in_intervened_graph(["Z"], ["Y"], ["X"]))

    def test_discrete_iv_random(self):
        iv_input = "Z -> X, X -> Y, U1 -> X, U1 -> Y, U2 -> Z"
        iv_cardinalities = {"Z": 4, "X": 3, "Y": 2, "U1": 0, "U2": 0}
        iv_unobs = ["U1", "U2"]
        iv_target = "Y"
        iv_target_value = 1
        iv_intervention = "X"
        iv_intervention_value = 1
        rel_path = DataExamplesPaths.CSV_DISCRETE_IV_RANDOM_EXAMPLE.value
        iv_csv_path = os.path.join(PROJECT_ROOT, rel_path)
        iv_df = pd.read_csv(iv_csv_path)

        model = CausalModel(
            data=iv_df,
            edges=iv_input,
            custom_cardinalities=iv_cardinalities,
            unobservables_labels=iv_unobs,
            interventions=(iv_intervention, iv_intervention_value),
            target=(iv_target, iv_target_value),
        )

        self.assertFalse(model.are_d_separated_in_complete_graph(["Z"], ["Y"], ["X"]))
        expected_lower, expected_upper = (0.17221135029354206, 0.8160779537149818)
        expected_lower = round(expected_lower, 3)
        expected_upper = round(expected_upper, 3)
        lower, upper = model.inference_intervention_query()
        lower = round(float(lower), 3)
        upper = round(float(upper), 3)
        self.assertEqual(lower, expected_lower)
        self.assertEqual(upper, expected_upper)

    def test_binary_copilot_example(self):
        copilot_input = "X -> Y, X -> D, D -> Y, E -> D, U1 -> Y, U1 -> X, U2 -> D, U3 -> E, U1 -> F"
        copilot_cardinalities = {
            "X": 2,
            "Y": 2,
            "D": 2,
            "E": 2,
            "F": 2,
            "U1": 0,
            "U2": 0,
            "U3": 0,
        }
        copilot_unobs = ["U1", "U2", "U3"]
        copilot_target = "Y"
        copilot_intervention = "X"
        rel_path = DataExamplesPaths.CSV_COPILOT_EXAMPLE.value
        copilot_csv_path = os.path.join(PROJECT_ROOT, rel_path)
        copilot_df = pd.read_csv(copilot_csv_path)

        model = CausalModel(
            data=copilot_df,
            edges=copilot_input,
            custom_cardinalities=copilot_cardinalities,
            unobservables_labels=copilot_unobs,
            interventions=(copilot_intervention, 1),
            target=(copilot_target, 1),
        )
        self.assertFalse(model.are_d_separated_in_complete_graph(["E"], ["X"], ["D"]))
        expected_lower, expected_upper = (0.3570355041940286, 0.8560355041940286)
        expected_lower = round(expected_lower, 3)
        expected_upper = round(expected_upper, 3)
        lower, upper = model.inference_intervention_query()
        lower = round(float(lower), 3)
        upper = round(float(upper), 3)
        self.assertEqual(lower, expected_lower)
        self.assertEqual(upper, expected_upper)
        model.set_interventions([("D", 1)])
        self.assertTrue(model.are_d_separated_in_intervened_graph(["E"], ["Y"], ["D"]))

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
        lower, upper = model.inference_intervention_query()
        lower = round(float(lower), 3)
        upper = round(float(upper), 3)
        self.assertEqual(lower, expected_lower)
        self.assertEqual(upper, expected_upper)

    def test_simple_counfoundness(self):
        edges = "U1 -> X, U1 -> Y"
        unobs = ["U1"]
        rel_path = DataExamplesPaths.CSV_BALKE_PEARL_EXAMPLE.value
        csv_path = os.path.join(PROJECT_ROOT, rel_path)
        df = pd.read_csv(csv_path)

        model = CausalModel(
            data=df,
            edges=edges,
            unobservables_labels=unobs,
        )

        self.assertTrue(model.are_d_separated_in_complete_graph(["X"], ["Y"], ["U1"]))

    def test_incident(self):
        edges_list_2 = [
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
            # UNOBS
            ("Unob_helper_1", "DB_Change"),
            ("Unob_helper_2", "DB_Latency"),
            ("Unob_helper_3", "MS-B_Error"),
            ("Unob_helper_4", "MS-A_Error"),
            ("Unob_helper_5", "MS-A_Threads"),
            ("Unob_helper_6", "MS-A_Crash"),
            ("Unob_helper_7", "Outage"),
        ]

        latent_nodes_2 = [
            "HeavyTraffic",
            "Unob_helper_1",
            "Unob_helper_2",
            "Unob_helper_3",
            "Unob_helper_4",
            "Unob_helper_5",
            "Unob_helper_6",
            "Unob_helper_7",
        ]
        edges_2 = nx.DiGraph(edges_list_2)
        rel_path = DataExamplesPaths.NEW_MEDIUM_SCALE_OUTAGE_INCIDENT.value
        csv_path = os.path.join(PROJECT_ROOT, rel_path)
        df_medium_scale_incident = pd.read_csv(csv_path, index_col=0)
        model = CausalModel(
            data=df_medium_scale_incident,
            edges=edges_2,
            unobservables_labels=latent_nodes_2,
        )
        self.assertTrue(
            model.are_d_separated_in_complete_graph(
                ["MS-A_Latency"], ["DB_Change"], ["DB_Latency"]
            )
        )

    def test_bcause_1(self):

        edges = "X -> Y, U -> X, U -> Y"
        unobs = ["U"]
        rel_path = DataExamplesPaths.CSV_BCAUSE_EXAMPLE_1.value
        csv_path = os.path.join(PROJECT_ROOT, rel_path)
        df = pd.read_csv(csv_path)
        bcause = {0: [0.12651440458851518, 0.41206478674401825], 
                  1: [0.4153088975671237, 0.7948002669191389]}

        for intervention_value in [0, 1]:
            with self.subTest(intervention_value=intervention_value):

                model = CausalModel(
                    data=df,
                    edges=edges,
                    unobservables_labels=unobs
                )

                lower, upper = model.inference_intervention_query(
                    interventions=[("X", intervention_value)],
                    target=("Y", 1)
                )
                tv = model.identifiable_intervention_query(
                    interventions=[("X", intervention_value)],
                    target=("Y", 1)
                )
                self.assertGreaterEqual(
                    float(upper),
                    tv,
                    msg=f"upper bound too low for do(X={intervention_value}) on Y=1"
                )
                self.assertLessEqual(
                    float(lower),
                    tv,
                    msg=f"lower bound too high for do(X={intervention_value}) on Y=1"
                )
                self.assertAlmostEqual(
                    float(lower),
                    bcause[intervention_value][0],
                    delta=0.3,
                    msg=f"lower bound does not match for do(X={intervention_value}) on Y=1"
                )
                self.assertAlmostEqual(
                    float(upper),
                    bcause[intervention_value][1],
                    delta=0.3,
                    msg=f"upper bound does not match for do(X={intervention_value}) on Y=1"
                )

    def test_bcause_2(self):

        edges = "X -> Y, U -> X, U -> Y, X -> W, W -> Y, U2 -> W"
        unobs = ["U", "U2"]
        rel_path = DataExamplesPaths.CSV_BCAUSE_EXAMPLE_2.value
        csv_path = os.path.join(PROJECT_ROOT, rel_path)
        df = pd.read_csv(csv_path)
        df["U2"] = np.random.binomial(1, 0.5, size=len(df))
        bcause = {0: [0.3991619515659455, 0.39916195179153174],
                  1: [0.7014971028665108, 0.7014971031112013]}

        for intervention_value in [0, 1]:
            with self.subTest(intervention_value=intervention_value):
                
                model_2 = CausalModel(
                    data=df,
                    edges=edges,
                    unobservables_labels=unobs
                )

                lower, upper = model_2.inference_intervention_query(
                    interventions=[("X", intervention_value)],
                    target=("Y", 1)
                )
                tv = model_2.identifiable_intervention_query(
                    interventions=[("X", intervention_value)],
                    target=("Y", 1)
                )
                self.assertGreaterEqual(
                    float(upper),
                    tv,
                    msg=f"upper bound too low for do(X={intervention_value}) on Y=1"
                )
                self.assertLessEqual(
                    float(lower),
                    tv,
                    msg=f"lower bound too high for do(X={intervention_value}) on Y=1"
                )
                self.assertAlmostEqual(
                    float(lower),
                    bcause[intervention_value][0],
                    delta=0.3,
                    msg=f"lower bound does not match for do(X={intervention_value}) on Y=1"
                )
                self.assertAlmostEqual(
                    float(upper),
                    bcause[intervention_value][1],
                    delta=0.3,
                    msg=f"upper bound does not match for do(X={intervention_value}) on Y=1"
                )


if __name__ == "__main__":
    unittest.main()
