from collections import namedtuple

import pandas as pd

dictAndIndex = namedtuple("dictAndIndex", ["mechanisms", "index"])


class ProbabilitiesHelper:
    """
    Common methods used to create the optimization problem.
    """

    def find_conditional_probability(
        dataFrame,
        indexToLabel,
        targetRealization: dict[int, int],
        conditionRealization: dict[int, int],
        v=True,
    ):
        """
        dataFrame              : pandas dataFrama that contains the data from the csv
        indexToLabel           : dictionary that converts an endogenous variable index to its label
        targetRealization      : specifies the values assumed by the endogenous variables V
        conditionalRealization : specifies the values assumed by the c-component tail T

        Calculates: P(V|T) = P(V,T) / P(T)
        """
        conditionProbability = ProbabilitiesHelper.find_probability(
            dataFrame, indexToLabel, conditionRealization, False
        )

        if conditionProbability == 0:
            return 0

        targetAndConditionProbability = ProbabilitiesHelper.find_probability(
            dataFrame, indexToLabel, targetRealization | conditionRealization, False)
        return targetAndConditionProbability / conditionProbability

    def find_probability(
        dataFrame, indexToLabel, variableRealizations: dict[int, int], v=True
    ):
        conditions = pd.Series([True] * len(dataFrame), index=dataFrame.index)
        for variable in variableRealizations:
            conditions &= (dataFrame[indexToLabel[variable]]
                           == variableRealizations[variable])

        compatibleCasesCount = dataFrame[conditions].shape[0]
        if v:
            print(f"Count compatible cases: {compatibleCasesCount}")
            print(f"Total cases: {dataFrame.shape[0]}")

        return compatibleCasesCount / dataFrame.shape[0]
