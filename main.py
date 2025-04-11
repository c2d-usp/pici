import pandas as pd

from causal_reasoning.causal_model import CausalModel
from causal_reasoning.utils._enum import Examples


def main():

    balke_input = "Z -> X, X -> Y, U1 -> X, U1 -> Y, U2 -> Z"
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
        unobservables_labels=balke_unobs,
        interventions=(balke_intervention, balke_intervention_value),
        target=(balke_target, balke_target_value),
    )

    print(balke_model.are_d_separated(["Z"], ["Y"], ["X"]))
    balke_model.inference_query()

    itau_input = (
        "X -> Y, X -> D, D -> Y, E -> D, U1 -> Y, U1 -> X, U2 -> D, U3 -> E, U1 -> F"
    )
    itau_unobs = ["U1", "U2", "U3"]
    itau_target = "Y"
    itau_intervention = "X"
    itau_csv_path = Examples.CSV_ITAU_EXAMPLE.value
    itau_df = pd.read_csv(itau_csv_path)

    itau_model = CausalModel(
        data=itau_df,
        edges=itau_input,
        unobservables_labels=itau_unobs,
        interventions=(itau_intervention, 1),
        target=(itau_target, 1),
    )
    itau_model.inference_query()


if __name__ == "__main__":
    main()
