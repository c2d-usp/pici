import pandas as pd
import logging

logger = logging.getLogger(__name__)


from causal_reasoning.graph.node import Node


def find_conditional_probability(
    dataFrame: pd.DataFrame,
    targetRealization: list[Node],
    conditionRealization: list[Node],
):
    """
    dataFrame              : pandas dataFrama that contains the data from the csv
    targetRealization      : specifies the values assumed by the endogenous variables V
    conditionalRealization : specifies the values assumed by the c-component tail T

    Calculates: P(V|T) = P(V,T) / P(T)

        Calculates: P(Target|Condition) = P(Target,Condition) / P(Condition)
    """
    conditionProbability = find_probability(dataFrame, conditionRealization)

    if conditionProbability == 0:
        return 0

    targetAndConditionRealization = targetRealization + conditionRealization

    targetAndConditionProbability = find_probability(
        dataFrame, targetAndConditionRealization
    )
    return targetAndConditionProbability / conditionProbability


def find_probability(dataFrame: pd.DataFrame, variables: list[Node]):
    compatibleCasesCount = count_occurrences(dataFrame, variables)
    totalCases = dataFrame.shape[0]
    logger.debug(f"Count compatible cases: {compatibleCasesCount}")
    logger.debug(f"Total cases: {totalCases}")
    return compatibleCasesCount / totalCases


def count_occurrences(dataFrame: pd.DataFrame, variables: list[Node]):
    conditions = pd.Series([True] * len(dataFrame), index=dataFrame.index)
    for variable_node in variables:
        conditions &= dataFrame[variable_node.label] == variable_node.value

    return dataFrame[conditions].shape[0]
