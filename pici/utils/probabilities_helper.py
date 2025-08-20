import logging

import pandas as pd

logger = logging.getLogger(__name__)


from pici.graph.node import Node


def find_conditional_probability(
    dataFrame: pd.DataFrame,
    target_realization: list[Node],
    condition_realization: list[Node],
):
    """
    dataFrame              : pandas dataFrama that contains the data from the csv
    target_realization      : specifies the values assumed by the endogenous variables V
    conditional_realization : specifies the values assumed by the c-component tail T

    Calculates: P(V|T) = P(V,T) / P(T)

        Calculates: P(Target|Condition) = P(Target,Condition) / P(Condition)
    """
    conditionProbability = find_probability(dataFrame, condition_realization)

    if conditionProbability == 0:
        return 0

    target_and_condition_realization = target_realization + condition_realization

    target_and_condition_probability = find_probability(
        dataFrame, target_and_condition_realization
    )
    return target_and_condition_probability / conditionProbability


def find_probability(dataFrame: pd.DataFrame, variables: list[Node]):
    compatible_cases_count = count_occurrences(dataFrame, variables)
    total_cases = dataFrame.shape[0]
    logger.debug(f"Count compatible cases: {compatible_cases_count}")
    logger.debug(f"Total cases: {total_cases}")
    return compatible_cases_count / total_cases


def count_occurrences(dataFrame: pd.DataFrame, variables: list[Node]):
    conditions = pd.Series([True] * len(dataFrame), index=dataFrame.index)
    for variable_node in variables:
        conditions &= dataFrame[variable_node.label] == variable_node.value
    return dataFrame[conditions].shape[0]


def find_conditional_probability2(
    dataFrame: pd.DataFrame,
    targetRealization: dict[str, int],
    conditionRealization: dict[str, int],
):
    """
    dataFrame              : pandas dataFrama that contains the data from the csv
    targetRealization      : specifies the values assumed by the endogenous variables V
    conditionalRealization : specifies the values assumed by the c-component tail T

    Calculates: P(V|T) = P(V,T) / P(T)

        Calculates: P(Target|Condition) = P(Target,Condition) / P(Condition)
    """
    conditionProbability = find_probability2(dataFrame, conditionRealization)

    if conditionProbability == 0:
        return 0

    targetAndConditionRealization = targetRealization | conditionRealization

    targetAndConditionProbability = find_probability2(
        dataFrame, targetAndConditionRealization
    )
    return targetAndConditionProbability / conditionProbability


def find_probability2(dataFrame: pd.DataFrame, realizationDict: dict[str, int]):
    compatibleCasesCount = count_occurrences_2(dataFrame, realizationDict)
    totalCases = dataFrame.shape[0]

    return compatibleCasesCount / totalCases


def count_occurrences_2(dataFrame: pd.DataFrame, realizationDict: dict[str, int]):
    conditions = pd.Series([True] * len(dataFrame), index=dataFrame.index)
    for variable_str in realizationDict:
        conditions &= dataFrame[variable_str] == realizationDict[variable_str]

    return dataFrame[conditions].shape[0]
