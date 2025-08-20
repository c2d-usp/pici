from collections import namedtuple
import itertools
import logging

logger = logging.getLogger(__name__)


from pici.graph.node import Node

dictAndIndex = namedtuple("dictAndIndex", ["mechanisms", "index"])


class MechanismGenerator:
    def helper_generate_spaces(nodes: list[Node]):
        spaces: list[list[int]] = []
        for node in nodes:
            spaces.append(range(0, node.cardinality))
        return spaces

    def generate_cross_products(list_spaces: list[list[int]]):
        cross_products_tuples = itertools.product(*list_spaces)
        return [list(combination) for combination in cross_products_tuples]

    def mechanisms_generator(
        latent_node: Node,
        endogenous_nodes: list[Node],
    ):
        """
        Generates an enumeration (list) of all mechanism a latent value can assume in its c-component. The c-component has to have
        exactly one latent variable.

        latent_node: an identifier for the latent node of the c-component
        endogenous_nodes: list of endogenous node of the c-component
        PS: Note that some parents may not be in the c-component, but the ones in the tail are also necessary for this function, so they
        must be included.

        """
        aux_spaces: list[list[int]] = []
        header_array: list[str] = []
        all_cases_list: list[list[list[int]]] = []
        dict_keys: list[str] = []

        for endogenous_node in endogenous_nodes:
            aux_spaces.clear()
            header: str = f"determines variable: {endogenous_node.label}"
            amount: int = 1
            ordered_parents: list[Node] = []
            for parent in endogenous_node.parents:
                if parent.label != latent_node.label:
                    ordered_parents.append(parent)
                    header = f"{parent.label}, " + header
                    aux_spaces.append(range(parent.cardinality))
                    amount *= parent.cardinality

            header_array.append(header + f" (x {amount})")
            logger.debug(f"auxSpaces {aux_spaces}")
            function_domain: list[list[int]] = [
                list(aux_tuple) for aux_tuple in itertools.product(*aux_spaces)
            ]
            logger.debug(f"functionDomain {function_domain}")

            image_values: list[int] = range(endogenous_node.cardinality)

            var_result = [
                [domain_case + [c] for c in image_values] for domain_case in function_domain
            ]
            logger.debug(f"For variable {endogenous_node.label}:")
            logger.debug(f"Function domain: {function_domain}")
            logger.debug(f"VarResult: {var_result}")

            for domain_case in function_domain:
                current_key = []
                for index, el in enumerate(domain_case):
                    current_key.append(f"{ordered_parents[index].label}={el}")
                key: str = ""
                for e in sorted(current_key):
                    key += f"{e},"
                dict_keys.append(key[:-1])

            all_cases_list = all_cases_list + var_result

        logger.debug(header_array)
        logger.debug(
            f"List all possible mechanism, placing in the same array those that determine the same function:\n{all_cases_list}"
        )
        logger.debug(
            f"List the keys of the dictionary (all combinations of the domains of the functions): {dict_keys}"
        )

        all_possible_mechanisms = list(itertools.product(*all_cases_list))
        mechanism_dicts: list[dict[str, int]] = []
        for index, mechanism in enumerate(all_possible_mechanisms):
            logger.debug(f"{index}) {mechanism}")
            curr_dict: dict[str, int] = {}
            for domain_index, node_function in enumerate(mechanism):
                logger.debug(f"The node function = {node_function}")
                curr_dict[dict_keys[domain_index]] = node_function[-1]

            mechanism_dicts.append(curr_dict)

        logger.debug("Check if the mechanism dictionary is working as expected:")
        for mechanism_dict in mechanism_dicts:
            for key in mechanism_dict:
                logger.debug(f"key: {key} & val: {mechanism_dict[key]} ")
            logger.debug("------------")

        """
        mechanism_dicts: list[dict[str, int]]
        --- Has all the mechanisms for ONE latent variable. Each element of the list is a set of mechanisms, which specify
            the value of any c-component endogenous node given the values of its endogenous parents.

        --- The key to check how one node behaves given its parents is a string with the value of the parents:
            "Parent1=Val1,Parent2=Val2,...,ParentN=ValN"

        --- There is an specific order for the parents: it is the same as in graph.graphNodes.
        """
        return all_possible_mechanisms, dict_keys, mechanism_dicts
