from enum import Enum


# All the paths are from the project root directory
class DirectoriesPath(Enum):
    CSV_PATH = "causal_reasoning/data/csv/"
    IMAGES_PATH = "causal_reasoning/data/images/"


class DataExamplesPaths(Enum):
    CSV_COPILOT_EXAMPLE = "causal_reasoning/data/csv/copilot.csv"
    CSV_BALKE_PEARL_EXAMPLE = "causal_reasoning/data/csv/balke_pearl.csv"
    CSV_DISCRETE_IV_RANDOM_EXAMPLE = (
        "causal_reasoning/data/csv/random_probabilities.csv"
    )
    CSV_N1M1 = "causal_reasoning/data/csv/n1_m1_scaling_case.csv"
    CSV_N2M1 = "causal_reasoning/data/csv/n2_m1_scaling_case.csv"
    CSV_N3M1 = "causal_reasoning/data/csv/n3_m1_scaling_case.csv"
    CSV_N4M1 = "causal_reasoning/data/csv/n4_m1_scaling_case.csv"
    CSV_N5M1 = "causal_reasoning/data/csv/n5_m1_scaling_case.csv"
    CSV_N1M2 = "causal_reasoning/data/csv/n1_m2_scaling_case.csv"
    CSV_N2M2 = "causal_reasoning/data/csv/n2_m2_scaling_case.csv"
    CSV_N3M2 = "causal_reasoning/data/csv/n3_m2_scaling_case.csv"
    CSV_N4M2 = "causal_reasoning/data/csv/n4_m2_scaling_case.csv"
    CSV_N5M2 = "causal_reasoning/data/csv/n5_m2_scaling_case.csv"
    CSV_N1M3 = "causal_reasoning/data/csv/n1_m3_scaling_case.csv"
    CSV_N2M3 = "causal_reasoning/data/csv/n2_m3_scaling_case.csv"
    CSV_N3M3 = "causal_reasoning/data/csv/n3_m3_scaling_case.csv"
    CSV_N4M3 = "causal_reasoning/data/csv/n4_m3_scaling_case.csv"
    CSV_N5M3 = "causal_reasoning/data/csv/n5_m3_scaling_case.csv"
    CSV_N6M1 = "causal_reasoning/data/csv/n6_m1_scaling_case.csv"
    CSV_N7M1 = "causal_reasoning/data/csv/n7_m1_scaling_case.csv"
    CSV_N8M1 = "causal_reasoning/data/csv/n8_m1_scaling_case.csv"
    CSV_N9M1 = "causal_reasoning/data/csv/n9_m1_scaling_case.csv"
    CSV_N10M1 = "causal_reasoning/data/csv/n10_m1_scaling_case.csv"
    CSV_N1M4 = "causal_reasoning/data/csv/n1_m4_scaling_case.csv"
    CSV_N2M4 = "causal_reasoning/data/csv/n2_m4_scaling_case.csv"
    CSV_N3M4 = "causal_reasoning/data/csv/n3_m4_scaling_case.csv"
    CSV_N4M4 = "causal_reasoning/data/csv/n4_m4_scaling_case.csv"
    CSV_N5M4 = "causal_reasoning/data/csv/n5_m4_scaling_case.csv"
    CSV_N1M6 = "causal_reasoning/data/csv/n1_m6_scaling_case.csv"
    CSV_N2M6 = "causal_reasoning/data/csv/n2_m6_scaling_case.csv"
    CSV_N1M5 = "causal_reasoning/data/csv/n1_m5_scaling_case.csv"
    CSV_N2M5 = "causal_reasoning/data/csv/n2_m5_scaling_case.csv"


class PlotGraphColors(Enum):
    INTERVENTIONS = "yellow"
    TARGETS = "orange"
    UNOBSERVABLES = "lightgray"
    OBSERVABLES = "lightblue"
