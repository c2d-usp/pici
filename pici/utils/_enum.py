from enum import Enum, auto


class DirectoriesPath(Enum):
    CSV_PATH = "data/csv/"
    IMAGES_PATH = "data/images/"


class DataExamplesPaths(Enum):
    CSV_COPILOT_EXAMPLE = "data/csv/copilot.csv"
    CSV_BALKE_PEARL_EXAMPLE = "data/csv/balke_pearl.csv"
    CSV_DISCRETE_IV_RANDOM_EXAMPLE = "data/csv/random_probabilities.csv"
    CSV_BCAUSE_EXAMPLE_1 = "data/csv/bcause_example_1.csv"
    CSV_BCAUSE_EXAMPLE_2 = "data/csv/bcause_example_2.csv"
    CSV_N1M1 = "data/csv/n1_m1_scaling_case.csv"
    CSV_N2M1 = "data/csv/n2_m1_scaling_case.csv"
    CSV_N3M1 = "data/csv/n3_m1_scaling_case.csv"
    CSV_N4M1 = "data/csv/n4_m1_scaling_case.csv"
    CSV_N5M1 = "data/csv/n5_m1_scaling_case.csv"
    CSV_N6M1 = "data/csv/n6_m1_scaling_case.csv"
    CSV_N1M2 = "data/csv/n1_m2_scaling_case.csv"
    CSV_N2M2 = "data/csv/n2_m2_scaling_case.csv"
    CSV_N3M2 = "data/csv/n3_m2_scaling_case.csv"
    CSV_N4M2 = "data/csv/n4_m2_scaling_case.csv"
    CSV_N5M2 = "data/csv/n5_m2_scaling_case.csv"
    CSV_N1M3 = "data/csv/n1_m3_scaling_case.csv"
    CSV_N2M3 = "data/csv/n2_m3_scaling_case.csv"
    CSV_N3M3 = "data/csv/n3_m3_scaling_case.csv"
    CSV_N4M3 = "data/csv/n4_m3_scaling_case.csv"
    CSV_N1M4 = "data/csv/n1_m4_scaling_case.csv"
    CSV_N2M4 = "data/csv/n2_m4_scaling_case.csv"
    CSV_N3M4 = "data/csv/n3_m4_scaling_case.csv"
    CSV_N1M5 = "data/csv/n1_m5_scaling_case.csv"
    CSV_N2M5 = "data/csv/n2_m5_scaling_case.csv"
    CSV_N1M6 = "data/csv/n1_m6_scaling_case.csv"
    NEW_MEDIUM_SCALE_OUTAGE_INCIDENT = (
        "data/csv/new_medium_scale_outage_incident_seed42.csv"
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

class ColumnGenerationParameters(Enum):
    BIG_M = 1e4
    MAX_ITERACTIONS_ALLOWED = 2000
