import pandas as pd


def find_conditional_probability(
    dataFrame,
    indexToLabel,
    targetRealization: dict[int, int],
    conditionRealization: dict[int, int],
):
    """
    dataFrame              : pandas dataFrama that contains the data from the csv
    indexToLabel           : dictionary that converts an endogenous variable index to its label
    targetRealization      : specifies the values assumed by the endogenous variables V
    conditionalRealization : specifies the values assumed by the c-component tail T

    Calculates: P(V|T) = P(V,T) / P(T)

        Calculates: P(Target|Condition) = P(Target,Condition) / P(Condition)
    """
    conditionProbability = find_probability(
        dataFrame, indexToLabel, conditionRealization
    )

    if conditionProbability == 0:
        return 0

    targetAndConditionRealization = targetRealization | conditionRealization

    targetAndConditionProbability = find_probability(
        dataFrame, indexToLabel, targetAndConditionRealization)
    return targetAndConditionProbability / conditionProbability

def find_probability(
    dataFrame, indexToLabel, variableRealizations: dict[int, int]
):
    compatibleCasesCount = count_occurrences(dataFrame, indexToLabel, variableRealizations)
    totalCases = dataFrame.shape[0]
    # TODO: ADD LOG??
    if False:
        print(f"Count compatible cases: {compatibleCasesCount}")
        print(f"Total cases: {totalCases}")
    return compatibleCasesCount / totalCases

def count_occurrences(dataFrame: pd.DataFrame, indexToLabel: dict[int, str], variableRealizations: dict[int, int]):
    conditions = pd.Series([True] * len(dataFrame), index=dataFrame.index)
    for variable_key in variableRealizations:
        conditions &= (dataFrame[indexToLabel[variable_key]]
                        == variableRealizations[variable_key])

    return dataFrame[conditions].shape[0]
