from enum import Enum


# All the paths are from the project root directory
class DirectoriesPath(Enum):
    CSV_PATH = "causal_reasoning/data/"


class Examples(Enum):
    CSV_ITAU_EXAMPLE = "causal_reasoning/data/itau.csv"
    CSV_BALKE_PEARL_EXAMPLE = "causal_reasoning/data/balke_pearl.csv"
    CSV_DISCRETE_IV_RANDOM_EXAMPLE = "causal_reasoning/data/random_probabilities.csv"
    NEW_MEDIUM_SCALE_OUTAGE_INCIDENT="causal_reasoning/data/new_medium_scale_outage_incident_seed42.csv"
    CSV_N1M1="causal_reasoning/data/n1_m1_scaling_case.csv"
    CSV_N2M1="causal_reasoning/data/n2_m1_scaling_case.csv"
    CSV_N3M1="causal_reasoning/data/n3_m1_scaling_case.csv"
    CSV_N4M1="causal_reasoning/data/n4_m1_scaling_case.csv"
    CSV_N5M1="causal_reasoning/data/n5_m1_scaling_case.csv"    
    CSV_N6M1="causal_reasoning/data/n6_m1_scaling_case.csv"
    CSV_N1M2="causal_reasoning/data/n1_m2_scaling_case.csv"
    CSV_N2M2="causal_reasoning/data/n2_m2_scaling_case.csv"
    CSV_N3M2="causal_reasoning/data/n3_m2_scaling_case.csv"
    CSV_N4M2="causal_reasoning/data/n4_m2_scaling_case.csv"
    CSV_N5M2="causal_reasoning/data/n5_m2_scaling_case.csv"
    CSV_N1M3="causal_reasoning/data/n1_m3_scaling_case.csv"
    CSV_N2M3="causal_reasoning/data/n2_m3_scaling_case.csv"
    CSV_N3M3="causal_reasoning/data/n3_m3_scaling_case.csv"
    CSV_N4M3="causal_reasoning/data/n4_m3_scaling_case.csv"
    CSV_N1M4="causal_reasoning/data/n1_m4_scaling_case.csv"
    CSV_N2M4="causal_reasoning/data/n2_m4_scaling_case.csv"
    CSV_N3M4="causal_reasoning/data/n3_m4_scaling_case.csv"
    CSV_N1M5="causal_reasoning/data/n1_m5_scaling_case.csv"
    CSV_N2M5="causal_reasoning/data/n2_m5_scaling_case.csv"
    CSV_N1M6="causal_reasoning/data/n1_m6_scaling_case.csv"