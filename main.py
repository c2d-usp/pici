import pandas as pd
import logging

from causal_reasoning.causal_model import CausalModel
from causal_reasoning.utils._enum import Examples


def binary_balke_pearl_example():
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
        interventions=(balke_intervention, balke_intervention_value),
        target=(balke_target, balke_target_value),
    )

    # print(f">> Is Z d-separated from Y giving X? {balke_model.are_d_separated(['Z'], ['Y'], ['X'])}")
    balke_model.inference_intervention_query()


def discrete_iv_random():
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

    # print(
    #     f">> Is Z d-separated from Y giving X? {iv_model.are_d_separated(['Z'], ['Y'], ['X'])}"
    # )
    iv_model.inference_intervention_query()


def binary_itau_example():
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
    itau_model.inference_intervention_query()


def double_intervention():
    edges = "Z -> X, X -> Y, U1 -> X, U1 -> Y, U2 -> Z"
    cardinalities = {"Z": 2, "X": 2, "Y": 2, "U1": 0, "U2": 0}
    unobs = ["U1", "U2"]
    target = "Y"
    target_value = 1
    intervention = "X"
    intervention_value = 1
    csv_path = Examples.CSV_BALKE_PEARL_EXAMPLE.value
    df = pd.read_csv(csv_path)

    model = CausalModel(
        data=df,
        edges=edges,
        custom_cardinalities=cardinalities,
        unobservables_labels=unobs,
        interventions=[(intervention, intervention_value), ("Z",1)],
        target=(target, target_value),
    )

    print(f">> Is Z d-separated from Y giving X? {model.are_d_separated(['Z'], ['Y'], ['X'])}")
    model.inference_intervention_query()

def discrete_itau_example():
    raise NotImplementedError


def main():

    logging.basicConfig(level=logging.INFO)

    # binary_balke_pearl_example()
    # discrete_iv_random()
    # binary_itau_example()
    double_intervention()


if __name__ == "__main__":
    main()
