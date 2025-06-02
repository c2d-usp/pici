import unittest

import pandas as pd

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

        balke_model = CausalModel(
            data=balke_df,
            edges=balke_input,
            custom_cardinalities=balke_cardinalities,
            unobservables_labels=balke_unobs,
        )

        balke_model.set_interventions([(balke_intervention, balke_intervention_value)])
        balke_model.set_target((balke_target, balke_target_value))

        self.assertFalse(balke_model.are_d_separated(['Z'], ['Y'], ['X']))
        # balke_model.inference_intervention_query()


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

        iv_model = CausalModel(
            data=iv_df,
            edges=iv_input,
            custom_cardinalities=iv_cardinalities,
            unobservables_labels=iv_unobs,
            interventions=(iv_intervention, iv_intervention_value),
            target=(iv_target, iv_target_value),
        )

        self.assertFalse(iv_model.are_d_separated(['Z'], ['Y'], ['X']))
        # iv_model.inference_intervention_query()


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

        itau_model = CausalModel(
            data=itau_df,
            edges=itau_input,
            custom_cardinalities=itau_cardinalities,
            unobservables_labels=itau_unobs,
            interventions=(itau_intervention, 1),
            target=(itau_target, 1),
        )
        self.assertFalse(itau_model.are_d_separated(['E'], ['X'], ['D']))
        # itau_model.inference_intervention_query()


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

        self.assertFalse(model.are_d_separated(['Z'], ['Y'], ['X']))
        # model.inference_intervention_query()

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

        self.assertTrue(model.are_d_separated(['X'], ['Y'], ['U1']))

if __name__ == '__main__':
    unittest.main()