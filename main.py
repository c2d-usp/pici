import pandas as pd

from causal_usp_icti.utils._enum import Examples
from causal_usp_icti.causal_model import CausalModel


def main():
    balke_input = "Z -> X, X -> Y, U1 -> X, U1 -> Y, U2 -> Z"
    balke_unobs = ["U1", "U2"]
    balke_target = "Y"
    balke_intervention = "X"
    balke_csv_path = Examples.CSV_BALKE_PEARL_EXAMPLE.value
    balke_df = pd.read_csv(balke_csv_path)

    print("------")
    balke_model = CausalModel(
        data=balke_df,
        edges=balke_input,
        unobservables=balke_unobs,
        interventions=balke_intervention,
        interventions_value=1,
        target=balke_target,
        target_value=1,
    )
    balke_model.inference_query()

    itau_input = (
        "X -> Y, X -> D, D -> Y, E -> D, U1 -> Y, U1 -> X, U2 -> D, U3 -> E, U1 -> F"
    )
    itau_unobs = ["U1", "U2", "U3"]
    itau_target = "Y"
    itau_intervention = "X"
    itau_csv_path = Examples.CSV_ITAU_EXAMPLE.value
    itau_df = pd.read_csv(itau_csv_path)

    print("------")
    itau_model = CausalModel(
        data=itau_df,
        edges=itau_input,
        unobservables=itau_unobs,
        interventions=itau_intervention,
        interventions_value=1,
        target=itau_target,
        target_value=1,
    )
    itau_model.inference_query()


if __name__ == "__main__":
    main()
