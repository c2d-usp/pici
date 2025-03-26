import itertools
from collections import namedtuple

from causal_usp_icti.graph.node import Node

dictAndIndex = namedtuple("dictAndIndex", ["mechanisms", "index"])


class MechanismGenerator:
    def helper_generate_spaces(
            nodes: list[int], cardinalities: dict[int, int]):
        spaces: list[list[int]] = []

        for node in nodes:
            spaces.append(range(0, cardinalities[node]))

        return spaces

    def generate_cross_products(listSpaces: list[list[int]]):
        crossProductsTuples = itertools.product(*listSpaces)
        return [list(combination) for combination in crossProductsTuples]

    def mechanisms_generator(
        latentNode: int,
        endogenousNodes: list[int],
        cardinalities: dict[int, int],
        graphNodes: list[Node],
        v=True,
    ):
        """
        Generates an enumeration (list) of all mechanism a latent value can assume in its c-component. The c-component has to have
        exactly one latent variable.

        latentNode: an identifier for the latent node of the c-component
        endogenousNodes: list of endogenous node of the c-component
        cardinalities: dictionary with the cardinalities of the endogenous nodes. The key for each node is the number that represents
        it in the endogenousNode list
        parentsDict: dictionary that has the same key as the above argument, but instead returns a list with the parents of each endogenous
        node. PS: Note that some parents may not be in the c-component, but the ones in the tail are also necessary for this function, so they
        must be included.
        v (verbose): enable or disable the logs
        """

        auxSpaces: list[list[int]] = []
        headerArray: list[str] = []
        allCasesList: list[list[list[int]]] = []
        dictKeys: list[str] = []

        for var in endogenousNodes:
            auxSpaces.clear()
            header: str = f"determines variable: {var}"
            amount: int = 1
            orderedParents: list[int] = []
            for parent in graphNodes[var].parents:
                if parent != latentNode:
                    orderedParents.append(parent)
                    header = f"{parent}, " + header
                    auxSpaces.append(range(cardinalities[parent]))
                    amount *= cardinalities[parent]

            headerArray.append(header + f" (x {amount})")
            functionDomain: list[list[int]] = [
                list(auxTuple) for auxTuple in itertools.product(*auxSpaces)
            ]
            if v:
                print(functionDomain)

            imageValues: list[int] = range(cardinalities[var])

            varResult = [[domainCase + [c] for c in imageValues]
                         for domainCase in functionDomain]
            if v:
                print(f"For variable {var}:")
                print(f"Function domain: {functionDomain}")
                print(f"VarResult: {varResult}")

            for domainCase in functionDomain:
                key: str = ""
                for index, el in enumerate(domainCase):
                    key = key + f"{orderedParents[index]}={el},"
                dictKeys.append(key[:-1])

            allCasesList = allCasesList + varResult

        if v:
            print(headerArray)
            print(
                f"List all possible mechanism, placing in the same array those that determine the same function:\n{allCasesList}"
            )
            print(
                f"List the keys of the dictionary (all combinations of the domains of the functions): {dictKeys}"
            )

        allPossibleMechanisms = list(itertools.product(*allCasesList))
        mechanismDicts: list[dict[str, int]] = []
        for index, mechanism in enumerate(allPossibleMechanisms):
            if v:
                print(f"{index}) {mechanism}")
            currDict: dict[str, int] = {}
            for domainIndex, nodeFunction in enumerate(mechanism):
                if v:
                    print(f"The node function = {nodeFunction}")
                currDict[dictKeys[domainIndex]] = nodeFunction[-1]

            mechanismDicts.append(currDict)

        if v:
            print("Check if the mechanism dictionary is working as expected:")
            for mechanismDict in mechanismDicts:
                for key in mechanismDict:
                    print(f"key: {key} & val: {mechanismDict[key]}")
                print("------------")

        """
        mechanismDicts: list[dict[str, int]]
        --- Has all the mechanisms for ONE latent variable. Each element of the list is a set of mechanisms, which specify
            the value of any c-component endogenous node given the values of its endogenous parents.

        --- The key to check how one node behaves given its parents is a string with the value of the parents:
            "Parent1=Val1,Parent2=Val2,...,ParentN=ValN"

        --- There is an specific order for the parents: it is the same as in graph.graphNodes.

        TODO: change the dictionary so that it also depends explicitly on the node we want to determine the value, otherwise the
        key for two nodes with the same parents is the same.
        """

        return allPossibleMechanisms, dictKeys, mechanismDicts

    # Not used, but useful when there is more than one latent in the
    # optimization system
    def mechanism_list_generator(
        cardinalities: dict[int, int],
        listU: list[int],
        listSpaces: set[int],
        graphNodes: list[Node],
    ):
        mechanismDictsList: list[list[dictAndIndex]] = []
        globalIndex: int = 0
        latentCardinalities: dict[int, int] = {}
        for latentVariable in listU:
            endogenousInS: list[int] = list(
                set(graphNodes[latentVariable].children) & listSpaces
            )
            _, _, mechanismDicts = MechanismGenerator.mechanisms_generator(
                latentNode=latentVariable,
                endogenousNodes=endogenousInS,
                cardinalities=cardinalities,
                graphNodes=graphNodes,
                v=False,
            )

            mechanismIndexDict: list[dictAndIndex] = []
            initVal: int = globalIndex
            for mechanismDict in mechanismDicts:
                mechanismIndexDict.append(
                    dictAndIndex(mechanismDict, globalIndex))
                globalIndex += 1

            latentCardinalities[latentVariable] = globalIndex - initVal
            mechanismDictsList.append(mechanismIndexDict)

        return mechanismDictsList, latentCardinalities
