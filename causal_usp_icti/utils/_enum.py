from enum import Enum


# All the paths are from the project root directory
class DirectoriesPath(Enum):
    TEST_CASES_INPUTS = "causal_usp_icti/test_cases/inputs"
    CSV_PATH = "causal_usp_icti/data"


class Examples(Enum):
    CSV_ITAU_EXAMPLE = "causal_usp_icti/data/itau.csv"
    TXT_ITAU_EXAMPLE = "causal_usp_icti/test_cases/inputs/itau.txt"
    CSV_BALKE_PEARL_EXAMPLE = "causal_usp_icti/data/balke_pearl.csv"
    TXT_BALKE_PEARL_EXAMPLE = "causal_usp_icti/test_cases/inputs/balke_pearl.txt"
