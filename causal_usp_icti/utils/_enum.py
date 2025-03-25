from enum import Enum


# All the paths are from the project root directory
class DirectoriesPath(Enum):
    CSV_PATH = "causal_usp_icti/data/"


class Examples(Enum):
    CSV_ITAU_EXAMPLE = "causal_usp_icti/data/itau.csv"
    CSV_BALKE_PEARL_EXAMPLE = "causal_usp_icti/data/balke_pearl.csv"
