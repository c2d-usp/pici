from enum import Enum, auto


class DirectoriesPath(Enum):
    CSV_PATH = "pici/data/csv/"
    IMAGES_PATH = "pici/data/images/"


class DataExamplesPaths(Enum):
    CSV_COPILOT_EXAMPLE = "pici/data/csv/copilot.csv"
    CSV_BALKE_PEARL_EXAMPLE = "pici/data/csv/balke_pearl.csv"
    CSV_DISCRETE_IV_RANDOM_EXAMPLE = (
        "pici/data/csv/random_probabilities.csv"
    )
    CSV_BCAUSE_EXAMPLE_1 = "pici/data/csv/bcause_example_1.csv"
    CSV_BCAUSE_EXAMPLE_2 = "pici/data/csv/bcause_example_2.csv"
    CSV_N1M1 = "pici/data/csv/n1_m1_scaling_case.csv"
    CSV_N2M1 = "pici/data/csv/n2_m1_scaling_case.csv"
    CSV_N3M1 = "pici/data/csv/n3_m1_scaling_case.csv"
    CSV_N4M1 = "pici/data/csv/n4_m1_scaling_case.csv"
    CSV_N5M1 = "pici/data/csv/n5_m1_scaling_case.csv"
    CSV_N6M1 = "pici/data/csv/n6_m1_scaling_case.csv"
    CSV_N1M2 = "pici/data/csv/n1_m2_scaling_case.csv"
    CSV_N2M2 = "pici/data/csv/n2_m2_scaling_case.csv"
    CSV_N3M2 = "pici/data/csv/n3_m2_scaling_case.csv"
    CSV_N4M2 = "pici/data/csv/n4_m2_scaling_case.csv"
    CSV_N5M2 = "pici/data/csv/n5_m2_scaling_case.csv"
    CSV_N1M3 = "pici/data/csv/n1_m3_scaling_case.csv"
    CSV_N2M3 = "pici/data/csv/n2_m3_scaling_case.csv"
    CSV_N3M3 = "pici/data/csv/n3_m3_scaling_case.csv"
    CSV_N4M3 = "pici/data/csv/n4_m3_scaling_case.csv"
    CSV_N1M4 = "pici/data/csv/n1_m4_scaling_case.csv"
    CSV_N2M4 = "pici/data/csv/n2_m4_scaling_case.csv"
    CSV_N3M4 = "pici/data/csv/n3_m4_scaling_case.csv"
    CSV_N1M5 = "pici/data/csv/n1_m5_scaling_case.csv"
    CSV_N2M5 = "pici/data/csv/n2_m5_scaling_case.csv"
    CSV_N1M6 = "pici/data/csv/n1_m6_scaling_case.csv"
    NEW_MEDIUM_SCALE_OUTAGE_INCIDENT = (
        "pici/data/csv/new_medium_scale_outage_incident_seed42.csv"
    )


class PlotGraphColors(Enum):
    INTERVENTIONS = "yellow"
    TARGETS = "orange"
    UNOBSERVABLES = "lightgray"
    OBSERVABLES = "lightblue"


class OptimizersLabels(Enum):
    GUROBI = "gurobi"
    SCIPY = "scipy"


class OptimizationDirection(Enum):
    MINIMIZE = auto()
    MAXIMIZE = auto()


class GurobiParameters(Enum):
    OUTPUT_SUPRESSED = 0
    OUTPUT_VERBOSE = 1
    DefaultObjectiveCoefficients = 1
