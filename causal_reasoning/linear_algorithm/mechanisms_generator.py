import itertools
from collections import namedtuple

from causal_reasoning.graph.node import T, Node

dictAndIndex = namedtuple("dictAndIndex", ["mechanisms", "index"])


class MechanismGenerator:
    def helper_generate_spaces(
            nodes: list[Node]):
        spaces: list[list[int]] = []
        for node in nodes:
            spaces.append(range(0, node.cardinality))
        return spaces

    def generate_cross_products(listSpaces: list[list[int]]):
        crossProductsTuples = itertools.product(*listSpaces)
        return [list(combination) for combination in crossProductsTuples]

    def mechanisms_generator(
        latentNode: Node,
        endogenousNodes: list[Node],
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

        """
        verbose = False
        auxSpaces: list[list[int]] = []
        headerArray: list[str] = []
        allCasesList: list[list[list[int]]] = []
        dictKeys: list[str] = []

        for endogenous_node in endogenousNodes:
            auxSpaces.clear()
            header: str = f"determines variable: {endogenous_node.label}"
            amount: int = 1
            ordered_parents: list[Node] = []
            for parent in endogenous_node.parents:
                if parent.label != latentNode.label:
                    ordered_parents.append(parent)
                    header = f"{parent.label}, " + header
                    auxSpaces.append(range(parent.cardinality))
                    amount *= parent.cardinality

            headerArray.append(header + f" (x {amount})")
            if verbose:
                print(f'auxSpaces {auxSpaces}')
            functionDomain: list[list[int]] = [
                list(auxTuple) for auxTuple in itertools.product(*auxSpaces)
            ]
            if verbose:
                print(f'functionDomain {functionDomain}')

            imageValues: list[int] = range(endogenous_node.cardinality)

            varResult = [[domainCase + [c] for c in imageValues]
                         for domainCase in functionDomain]
            # TODO: LOGGING
            if verbose:
                print(f"For variable {endogenous_node.label}:")
                print(f"Function domain: {functionDomain}")
                print(f"VarResult: {varResult}")

            for domainCase in functionDomain:
                current_key = []
                for index, el in enumerate(domainCase):
                    current_key.append(f"{ordered_parents[index].label}={el}")
                key: str = ""
                for e in sorted(current_key):
                    key += f"{e},"
                dictKeys.append(key[:-1])

            allCasesList = allCasesList + varResult

        # TODO: LOGGING
        if verbose:
            print(headerArray)
            print(
                f"List all possible mechanism, placing in the same array those that determine the same function:\n{allCasesList}"
            )
            print(
                f"List the keys of the dictionary (all combinations of the domains of the functions): {dictKeys}"
            )

        allPossibleMechanisms = list(itertools.product(*allCasesList))
        mechanismDicts: list[dict[T, int]] = []
        for index, mechanism in enumerate(allPossibleMechanisms):
            # TODO: LOGGING
            if verbose:
                print(f"{index}) {mechanism}")
            currDict: dict[T, int] = {}
            for domainIndex, nodeFunction in enumerate(mechanism):
                # TODO: LOGGING
                if verbose:
                    print(f"The node function = {nodeFunction}")
                currDict[dictKeys[domainIndex]] = nodeFunction[-1]

            mechanismDicts.append(currDict)

        # TODO: LOGGING
        if verbose:
            print("Check if the mechanism dictionary is working as expected:")
            for mechanismDict in mechanismDicts:
                for key in mechanismDict:
                    print(f"key: {key} & val: {mechanismDict[key]}")
                print("------------")

        """
        mechanismDicts: list[dict[T, int]]
        --- Has all the mechanisms for ONE latent variable. Each element of the list is a set of mechanisms, which specify
            the value of any c-component endogenous node given the values of its endogenous parents.

        --- The key to check how one node behaves given its parents is a string with the value of the parents:
            "Parent1=Val1,Parent2=Val2,...,ParentN=ValN"

        --- There is an specific order for the parents: it is the same as in graph.graphNodes.
        """
        return allPossibleMechanisms, dictKeys, mechanismDicts

    # Not used, but useful when there is more than one latent in the
    # optimization system
    def mechanism_list_generator(
        listU: list[Node],
        listSpaces: set[Node],
    ):
        mechanismDictsList: list[list[dictAndIndex]] = []
        globalIndex: int = 0
        for latentVariable in listU:
            endogenousInS: list[Node] = list(
                set(latentVariable.children) & listSpaces
            )
            _, _, mechanismDicts = MechanismGenerator.mechanisms_generator(
                latentNode=latentVariable,
                endogenousNodes=endogenousInS,
            )

            mechanismIndexDict: list[dictAndIndex] = []
            initVal: int = globalIndex
            for mechanismDict in mechanismDicts:
                mechanismIndexDict.append(
                    dictAndIndex(mechanismDict, globalIndex))
                globalIndex += 1

            latentVariable.cardinality = globalIndex - initVal
            mechanismDictsList.append(mechanismIndexDict)

        return mechanismDictsList
