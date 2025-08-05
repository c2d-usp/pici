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

from causal_reasoning.causal_model import CausalModel
from causal_reasoning.utils._enum import DataExamplesPaths

from auxiliary import genGraph


class TestMNCases(unittest.TestCase):
    def test_intervention_queries_via_subtests(self):
        cases = [
            (1, 1, DataExamplesPaths.CSV_N1M1),
            (2, 1, DataExamplesPaths.CSV_N2M1),
            (3, 1, DataExamplesPaths.CSV_N3M1),
            (4, 1, DataExamplesPaths.CSV_N4M1),
            (1, 2, DataExamplesPaths.CSV_N1M2),
        ]
        # Skip cases that are too long to run
        skip_cases = {(3, 1), (4, 1), (1, 2)}
        unobs = ["U1", "U2", "U3"]
        interventions = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for N, M, csv_example in cases:
            with self.subTest(N=N, M=M):
                if (N, M) in skip_cases:
                    self.skipTest(f"Skipping N={N}, M={M} (too long to run)")
                edges = genGraph(N=N, M=M)
                df = pd.read_csv(os.path.join(PROJECT_ROOT, csv_example.value))
                df["U3"] = np.random.binomial(1, 0.5, size=len(df))

                model = CausalModel(
                    data=df,
                    edges=edges,
                    unobservables_labels=unobs,
                )

                self.assertFalse(
                    model.are_d_separated_in_complete_graph(["X"], ["Y"], unobs),
                    msg=f"d-separation failed for N={N}, M={M}",
                )

                for target_value, intervention_value in interventions:
                    with self.subTest(
                        N=N, M=M, target=target_value, intervention=intervention_value
                    ):
                        model.set_interventions([("X", intervention_value)])
                        model.set_target(("Y", target_value))
                        lower, upper = model.partially_identifiable_intervention_query()

                        tv = model.identifiable_intervention_query()

                        self.assertGreaterEqual(
                            float(upper),
                            tv,
                            msg=f"upper bound too low for N={N},M={M},Y={target_value},do(X={intervention_value})",
                        )
                        self.assertLessEqual(
                            float(lower),
                            tv,
                            msg=f"lower bound too high for N={N},M={M},Y={target_value},do(X={intervention_value})",
                        )

if __name__ == "__main__":
    unittest.main()
