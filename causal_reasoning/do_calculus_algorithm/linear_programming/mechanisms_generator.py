import itertools
import logging
from collections import namedtuple

logger = logging.getLogger(__name__)


from causal_reasoning.graph.node import Node

dictAndIndex = namedtuple("dictAndIndex", ["mechanisms", "index"])


class MechanismGenerator:
    def helper_generate_spaces(nodes: list[Node]):
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
        PS: Note that some parents may not be in the c-component, but the ones in the tail are also necessary for this function, so they
        must be included.

        """
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
            logger.debug(f"auxSpaces {auxSpaces}")
            functionDomain: list[list[int]] = [
                list(auxTuple) for auxTuple in itertools.product(*auxSpaces)
            ]
            logger.debug(f"functionDomain {functionDomain}")

            imageValues: list[int] = range(endogenous_node.cardinality)

            varResult = [
                [domainCase + [c] for c in imageValues] for domainCase in functionDomain
            ]
            logger.debug(f"For variable {endogenous_node.label}:")
            logger.debug(f"Function domain: {functionDomain}")
            logger.debug(f"VarResult: {varResult}")

            for domainCase in functionDomain:
                current_key = []
                for index, el in enumerate(domainCase):
                    current_key.append(f"{ordered_parents[index].label}={el}")
                key: str = ""
                for e in sorted(current_key):
                    key += f"{e},"
                dictKeys.append(key[:-1])

            allCasesList = allCasesList + varResult

        logger.debug(headerArray)
        logger.debug(
            f"List all possible mechanism, placing in the same array those that determine the same function:\n{allCasesList}"
        )
        logger.debug(
            f"List the keys of the dictionary (all combinations of the domains of the functions): {dictKeys}"
        )

        allPossibleMechanisms = list(itertools.product(*allCasesList))
        mechanismDicts: list[dict[str, int]] = []
        for index, mechanism in enumerate(allPossibleMechanisms):
            logger.debug(f"{index}) {mechanism}")
            currDict: dict[str, int] = {}
            for domainIndex, nodeFunction in enumerate(mechanism):
                logger.debug(f"The node function = {nodeFunction}")
                currDict[dictKeys[domainIndex]] = nodeFunction[-1]

            mechanismDicts.append(currDict)

        logger.debug("Check if the mechanism dictionary is working as expected:")
        for mechanismDict in mechanismDicts:
            for key in mechanismDict:
                logger.debug(f"key: {key} & val: {mechanismDict[key]} ")
            logger.debug("------------")

        """
        mechanismDicts: list[dict[str, int]]
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
            endogenousInS: list[Node] = list(set(latentVariable.children) & listSpaces)
            _, _, mechanismDicts = MechanismGenerator.mechanisms_generator(
                latentNode=latentVariable,
                endogenousNodes=endogenousInS,
            )

            mechanismIndexDict: list[dictAndIndex] = []
            initVal: int = globalIndex
            for mechanismDict in mechanismDicts:
                mechanismIndexDict.append(dictAndIndex(mechanismDict, globalIndex))
                globalIndex += 1

            latentVariable.cardinality = globalIndex - initVal
            mechanismDictsList.append(mechanismIndexDict)

        return mechanismDictsList
