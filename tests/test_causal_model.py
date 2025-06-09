import unittest

import pandas as pd
import networkx as nx

from causal_reasoning.causal_model import CausalModel
from causal_reasoning.utils._enum import Examples

class TestInferenceAlgorithm(unittest.TestCase):
    def test_binary_balke_pearl_example(self):
        balke_input = "Z -> X, X -> Y, U1 -> X, U1 -> Y, U2 -> Z"
        balke_cardinalities = {"Z": 2, "X": 2, "Y": 2, "U1": 0, "U2": 0}
        balke_unobs = ["U1", "U2"]
        balke_target = "Y"
        balke_target_value = 1
        balke_intervention = "X"
        balke_intervention_value = 1
        balke_csv_path = Examples.CSV_BALKE_PEARL_EXAMPLE.value
        balke_df = pd.read_csv(balke_csv_path)

        model = CausalModel(
            data=balke_df,
            edges=balke_input,
            custom_cardinalities=balke_cardinalities,
            unobservables_labels=balke_unobs,
        )

        model.set_interventions([(balke_intervention, balke_intervention_value)])
        model.set_target((balke_target, balke_target_value))

        self.assertFalse(model.are_d_separated_in_complete_graph(['Z'], ['Y'], ['X']))
        expected_lower, expected_upper = (0.45000000000000007, 0.5199999999999999)
        expected_lower = round(expected_lower, 3)
        expected_upper = round(expected_upper, 3)
        lower, upper = model.inference_intervention_query()
        lower = round(float(lower), 3)
        upper = round(float(upper), 3)
        self.assertEqual(lower, expected_lower)
        self.assertEqual(upper, expected_upper)
        self.assertTrue(model.are_d_separated_in_intervened_graph(['Z'], ['Y'], ['X']))


    def test_discrete_iv_random(self):
        iv_input = "Z -> X, X -> Y, U1 -> X, U1 -> Y, U2 -> Z"
        iv_cardinalities = {"Z": 4, "X": 3, "Y": 2, "U1": 0, "U2": 0}
        iv_unobs = ["U1", "U2"]
        iv_target = "Y"
        iv_target_value = 1
        iv_intervention = "X"
        iv_intervention_value = 1
        iv_csv_path = Examples.CSV_DISCRETE_IV_RANDOM_EXAMPLE.value
        iv_df = pd.read_csv(iv_csv_path)

        model = CausalModel(
            data=iv_df,
            edges=iv_input,
            custom_cardinalities=iv_cardinalities,
            unobservables_labels=iv_unobs,
            interventions=(iv_intervention, iv_intervention_value),
            target=(iv_target, iv_target_value),
        )

        self.assertFalse(model.are_d_separated_in_complete_graph(['Z'], ['Y'], ['X']))
        expected_lower, expected_upper =(0.17221135029354206, 0.8160779537149818)
        expected_lower = round(expected_lower, 3)
        expected_upper = round(expected_upper, 3)
        lower, upper = model.inference_intervention_query()
        lower = round(float(lower), 3)
        upper = round(float(upper), 3)
        self.assertEqual(lower, expected_lower)
        self.assertEqual(upper, expected_upper)


    def test_binary_itau_example(self):
        itau_input = (
            "X -> Y, X -> D, D -> Y, E -> D, U1 -> Y, U1 -> X, U2 -> D, U3 -> E, U1 -> F"
        )
        itau_cardinalities = {
            "X": 2,
            "Y": 2,
            "D": 2,
            "E": 2,
            "F": 2,
            "U1": 0,
            "U2": 0,
            "U3": 0,
        }
        itau_unobs = ["U1", "U2", "U3"]
        itau_target = "Y"
        itau_intervention = "X"
        itau_csv_path = Examples.CSV_ITAU_EXAMPLE.value
        itau_df = pd.read_csv(itau_csv_path)

        model = CausalModel(
            data=itau_df,
            edges=itau_input,
            custom_cardinalities=itau_cardinalities,
            unobservables_labels=itau_unobs,
            interventions=(itau_intervention, 1),
            target=(itau_target, 1),
        )
        self.assertFalse(model.are_d_separated_in_complete_graph(['E'], ['X'], ['D']))
        expected_lower, expected_upper =(0.3570355041940286, 0.8560355041940286)
        expected_lower = round(expected_lower, 3)
        expected_upper = round(expected_upper, 3)
        lower, upper = model.inference_intervention_query()
        lower = round(float(lower), 3)
        upper = round(float(upper), 3)
        self.assertEqual(lower, expected_lower)
        self.assertEqual(upper, expected_upper)
        model.set_interventions([('D', 1)])
        self.assertTrue(model.are_d_separated_in_intervened_graph(['E'], ['Y'], ['D']))

    def test_double_intervention_binary_balke_pearl(self):
        edges = "Z -> X, X -> Y, U1 -> X, U1 -> Y, U2 -> Z"
        cardinalities = {"Z": 2, "X": 2, "Y": 2, "U1": 0, "U2": 0}
        unobs = ["U1", "U2"]
        target = "Y"
        target_value = 1
        csv_path = Examples.CSV_BALKE_PEARL_EXAMPLE.value
        df = pd.read_csv(csv_path)

        model = CausalModel(
            data=df,
            edges=edges,
            custom_cardinalities=cardinalities,
            unobservables_labels=unobs,
            interventions=[("X", 1), ("Z", 1)],
            target=(target, target_value),
        )

        self.assertFalse(model.are_d_separated_in_complete_graph(['Z'], ['Y'], ['X']))
        expected_lower, expected_upper =(0.07680001854838515, 0.09309998464330864)
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
        csv_path = Examples.CSV_BALKE_PEARL_EXAMPLE.value
        df = pd.read_csv(csv_path)

        model = CausalModel(
            data=df,
            edges=edges,
            unobservables_labels=unobs,
        )

        self.assertTrue(model.are_d_separated_in_complete_graph(['X'], ['Y'], ['U1']))
    
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
        df_medium_scale_incident = pd.read_csv(
            Examples.NEW_MEDIUM_SCALE_OUTAGE_INCIDENT.value, index_col=0
        )
        model = CausalModel(
            data=df_medium_scale_incident,
            edges=edges_2,
            unobservables_labels=latent_nodes_2,
        )
        self.assertTrue(model.are_d_separated_in_complete_graph(['MS-A_Latency'], ['DB_Change'], ['DB_Latency']))

        # model_2.set_interventions([(intervention_1, 0)])
        # model_2.set_target((target, 0))
        # print(f"{intervention_1}: PN = {model_2.inference_intervention_query()}")

        # model_2.set_interventions([(intervention_1, 1)])
        # model_2.set_target((target, 1))
        # print(f"{intervention_1}: PS = {model_2.inference_intervention_query()}")

        # model_2.set_interventions([(intervention_2, 0)])
        # model_2.set_target((target, 0))
        # print(f"{intervention_2}: PN = {model_2.inference_intervention_query()}")

        # model_2.set_interventions([(intervention_2, 1)])
        # model_2.set_target((target, 1))
        # print(f"{intervention_2}: PS = {model_2.inference_intervention_query()}")

if __name__ == '__main__':
    unittest.main()