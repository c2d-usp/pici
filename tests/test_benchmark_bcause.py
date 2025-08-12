'''
    Benchmark with bcause (https://github.com/PGM-Lab/bcause).
    The goal is to validate our approach compared to bcause.
'''

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

from pici.causal_model import CausalModel
from pici.utils._enum import DataExamplesPaths

class TestBenchmarkBcauseInference(unittest.TestCase):
    def test_bcause_1(self):
        edges = "X -> Y, U -> X, U -> Y"
        unobs = ["U"]
        df = pd.read_csv(os.path.join(PROJECT_ROOT, DataExamplesPaths.CSV_BCAUSE_EXAMPLE_1.value))
        bcause = {0: [0.12651440458851518, 0.41206478674401825], 
                  1: [0.4153088975671237, 0.7948002669191389]}

        for intervention_value in [0, 1]:
            with self.subTest(intervention_value=intervention_value):

                model = CausalModel(
                    data=df,
                    edges=edges,
                    unobservables_labels=unobs
                )

                lower, upper = model.intervention_query(
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
        df = pd.read_csv(os.path.join(PROJECT_ROOT, DataExamplesPaths.CSV_BCAUSE_EXAMPLE_2.value))
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

                lower, upper = model_2.intervention_query(
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
