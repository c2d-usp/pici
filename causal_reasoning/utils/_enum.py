from enum import Enum


# All the paths are from the project root directory
class DirectoriesPath(Enum):
    CSV_PATH = "causal_reasoning/data/"


class Examples(Enum):
    CSV_ITAU_EXAMPLE = "causal_reasoning/data/itau.csv"
    CSV_BALKE_PEARL_EXAMPLE = "causal_reasoning/data/balke_pearl.csv"
    CSV_DISCRETE_IV_RANDOM_EXAMPLE = "causal_reasoning/data/random_probabilities.csv"
