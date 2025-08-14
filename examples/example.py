import logging

import networkx as nx
import pandas as pd

from pici.causal_model import CausalModel
from pici.utils._enum import DataExamplesPaths


def incident_model():
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
        DataExamplesPaths.NEW_MEDIUM_SCALE_OUTAGE_INCIDENT.value, index_col=0
    )
    model_2 = CausalModel(
        data=df_medium_scale_incident,
        edges=edges_2,
        unobservables_labels=latent_nodes_2,
    )
    intervention_1 = "MS-A_Latency"
    intervention_2 = "DB_Latency"
    target = "Outage"

    print(
        f"Intervention: {intervention_1} Target: {target} --> Weak-PN = {model_2.weak_pn_inference(intervention_label=intervention_1, target_label=target)}"
    )

    print(
        f"Intervention: {intervention_1} Target: {target} --> Weak-PS = {model_2.weak_ps_inference(intervention_label=intervention_1, target_label=target)}"
    )

    print(
        f"Intervention: {intervention_2} Target: {target} --> Weak-PN = {model_2.weak_pn_inference(intervention_label=intervention_2, target_label=target)}"
    )

    print(
        f"Intervention: {intervention_2} Target: {target} --> Weak-PS = {model_2.weak_ps_inference(intervention_label=intervention_2, target_label=target)}"
    )


def binary_balke_pearl_example():
    balke_input = "Z -> X, X -> Y, U1 -> X, U1 -> Y, U2 -> Z"
    balke_cardinalities = {"Z": 2, "X": 2, "Y": 2, "U1": 0, "U2": 0}
    balke_unobs = ["U1", "U2"]
    balke_target = "Y"
    balke_target_value = 1
    balke_intervention = "X"
    balke_intervention_value = 1
    balke_csv_path = DataExamplesPaths.CSV_BALKE_PEARL_EXAMPLE.value
    balke_df = pd.read_csv(balke_csv_path)

    balke_model = CausalModel(
        data=balke_df,
        edges=balke_input,
        custom_cardinalities=balke_cardinalities,
        unobservables_labels=balke_unobs,
        interventions=(balke_intervention, balke_intervention_value),
        target=(balke_target, balke_target_value),
    )

    print(
        f"Is Z d-separated from node Y given node X? {balke_model.are_d_separated_in_intervened_graph(['Z'], ['Y'], ['X'])}"
    )

    balke_model.generate_graph_image("balke.png")

    lower, upper = balke_model.intervention_query()
    print(
        f"{lower} <= P({balke_target}={balke_target_value}|do({balke_intervention}={balke_intervention_value})) <= {upper}"
    )


def discrete_iv_random():
    iv_input = "Z -> X, X -> Y, U1 -> X, U1 -> Y, U2 -> Z"
    iv_cardinalities = {"Z": 4, "X": 3, "Y": 2, "U1": 0, "U2": 0}
    iv_unobs = ["U1", "U2"]
    iv_target = "Y"
    iv_target_value = 1
    iv_intervention = "X"
    iv_intervention_value = 1
    iv_csv_path = DataExamplesPaths.CSV_DISCRETE_IV_RANDOM_EXAMPLE.value
    iv_df = pd.read_csv(iv_csv_path)

    iv_model = CausalModel(
        data=iv_df,
        edges=iv_input,
        custom_cardinalities=iv_cardinalities,
        unobservables_labels=iv_unobs,
        interventions=(iv_intervention, iv_intervention_value),
        target=(iv_target, iv_target_value),
    )

    lower, upper = iv_model.intervention_query()
    print(
        f"{lower} <= P({iv_target}={iv_target_value}|do({iv_intervention}={iv_intervention_value})) <= {upper}"
    )
    iv_model.generate_graph_image("discrete_iv.png")


def binary_copilot_example():
    copilot_input = (
        "X -> Y, X -> D, D -> Y, E -> D, U1 -> Y, U1 -> X, U2 -> D, U3 -> E, U1 -> F"
    )
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
    copilot_csv_path = DataExamplesPaths.CSV_COPILOT_EXAMPLE.value
    copilot_df = pd.read_csv(copilot_csv_path)

    copilot_model = CausalModel(
        data=copilot_df,
        edges=copilot_input,
        custom_cardinalities=copilot_cardinalities,
        unobservables_labels=copilot_unobs,
        interventions=(copilot_intervention, 1),
        target=(copilot_target, 1),
    )

    copilot_model.generate_graph_image("copilot.png")


def main():

    logging.basicConfig(level=logging.INFO)

    binary_balke_pearl_example()
    discrete_iv_random()
    binary_copilot_example()
    incident_model()


if __name__ == "__main__":
    main()
