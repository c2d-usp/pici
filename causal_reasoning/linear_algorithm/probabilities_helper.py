import pandas as pd


def find_conditional_probability(
    dataFrame,
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
    conditionProbability = find_probability(
        dataFrame, conditionRealization
    )

    if conditionProbability == 0:
        return 0

    targetAndConditionRealization = targetRealization | conditionRealization

    targetAndConditionProbability = find_probability(
        dataFrame, targetAndConditionRealization)
    return targetAndConditionProbability / conditionProbability

def find_probability(
    dataFrame, variableRealizations: dict[str, int]
):
    compatibleCasesCount = count_occurrences(dataFrame, variableRealizations)
    totalCases = dataFrame.shape[0]
    # TODO: ADD LOG??
    if False:
        print(f"Count compatible cases: {compatibleCasesCount}")
        print(f"Total cases: {totalCases}")
    return compatibleCasesCount / totalCases

def count_occurrences(dataFrame: pd.DataFrame, variableRealizations: dict[str, int]):
    conditions = pd.Series([True] * len(dataFrame), index=dataFrame.index)
    for variable_label in variableRealizations:
        conditions &= (dataFrame[variable_label]
                        == variableRealizations[variable_label])

    return dataFrame[conditions].shape[0]
