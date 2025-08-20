import pandas as pd

from pici.graph.graph import Graph
from pici.graph.node import Node
from pici.intervention_inference_algorithm.linear_programming.mechanisms_generator import (
    MechanismGenerator,
)
from pici.utils.probabilities_helper import find_conditional_probability
from pici.utils.types import MechanismType


def create_realization_string(
    parents: list[Node], realization: list[int], indexer_list: list[Node]
):
    """
    Creates a unique string index for a set of parent nodes and their values.

    Args:
        parents (list[Node]): The parent nodes to index.
        realization (list[int]): The list of values corresponding to all nodes in indexer_list.
        indexer_list (list[Node]): The list of all nodes whose values are in realization.

    Returns:
        str: A comma-separated string representing the parent labels and their assigned values, sorted by label.
    """
    current_index = []
    for parent in parents:
        current_index.append(
            str(parent.label) + "=" + str(realization[indexer_list.index(parent)])
        )
    index: str = ""
    for e in sorted(current_index):
        index += f"{e},"
    return index[:-1]


def generate_constraints(
    data: pd.DataFrame,
    dag: Graph,
    unob: Node,
    considered_c_comp: list[Node],
    mechanisms: MechanismType,
) -> tuple[list[float], list[list[int]]]:
    """
    Generates the empirical probability constraints and decision matrix for the linear program.

    Args:
        data (pd.DataFrame): The dataset containing observed variable values.
        dag (Graph): The causal graph.
        unob (Node): The unobservable (latent) node.
        considered_c_comp (list[Node]): Nodes in the considered c-component.
        mechanisms (MechanismType): Mechanism mappings for the model.

    Returns:
        tuple[list[float], list[list[int]]]:
            - List of empirical probabilities for each constraint.
            - Decision matrix as a list of lists of coefficients.
    """
    topo_order: list[Node] = dag.topological_order
    c_comp_order = get_c_component_in_reverse_topological_order(
        topo_order, unob, considered_c_comp
    )
    c_component_and_tail: list[Node] = find_c_component_and_tail_set(unob, c_comp_order)

    symbolical_constraints_probabilities, Wc = (
        get_symbolical_constraints_probabilities_and_wc(
            c_comp_order, c_component_and_tail, topo_order
        )
    )

    probs = calculate_constraints_empirical_probabilities(
        data=data,
        Wc=Wc,
        symbolical_constraints_probabilities=symbolical_constraints_probabilities,
    )

    decision_matrix = calculate_decision_matrix(
        mechanisms=mechanisms,
        Wc=Wc,
        considered_c_comp=considered_c_comp,
        unob=unob,
    )

    return probs, decision_matrix


def get_c_component_in_reverse_topological_order(
    topo_order: list[Node], unob: Node, considered_c_comp: list[Node]
) -> list[Node]:
    """
    Finds nodes in the considered c-component that have the unobservable node as a parent,
    ordered in reverse topological order.

    Args:
        topo_order (list[Node]): Topological order of all nodes in the graph.
        unob (Node): The unobservable (latent) parent node from the intervention.
        considered_c_comp (list[Node]): Nodes in the considered c-component.

    Returns:
        list[Node]: Nodes in the c-component in reverse topological order.
    """
    c_comp_order: list[Node] = []
    for node in topo_order:
        if (unob in node.parents) and (node in considered_c_comp):
            c_comp_order.append(node)
    c_comp_order.reverse()
    return c_comp_order


def find_c_component_and_tail_set(unob: Node, c_comp_order: list[Node]) -> list[Node]:
    """
    Finds the union of the c-component nodes and their parents (excluding the unobservable node).

    Args:
        unob (Node): The unobservable (latent) parent node from the intervention.
        c_comp_order (list[Node]): Nodes in the c-component.

    Returns:
        list[Node]: List of c-component nodes and their parents, ordered.
    """
    c_component_and_tail: list[Node] = c_comp_order.copy()
    for node in c_comp_order:
        for parent in node.parents:
            if parent not in c_component_and_tail and (parent != unob):
                c_component_and_tail.append(parent)
    return c_component_and_tail


def get_symbolical_constraints_probabilities_and_wc(
    c_comp_order: list[Node], c_component_and_tail: list[Node], topo_order: list[Node]
) -> tuple[list[dict[Node, list[Node]]], list[Node]]:
    """
    Determines the symbolic constraints for probabilities and the set Wc of variables present in constraints.
    Wc is a subset of the union c-Component and Tail; Wc is a list of all variables present in the constraints.

    Args:
        c_comp_order (list[Node]): Nodes in the c-component.
        c_component_and_tail (list[Node]): Nodes in the c-component and their parents.
        topo_order (list[Node]): Topological order of all nodes in the graph.

    Returns:
        tuple[list[dict[Node, list[Node]]], list[Node]]:
            - List of dictionaries mapping each node to its conditioning variables.
            - List of all variables present in the constraints (Wc).
    """
    cond_vars: list[Node] = []
    symbolical_constraints_probabilities: list[dict[Node, list[Node]]] = []
    Wc: list[Node] = []
    Wc = c_comp_order.copy()
    while bool(c_comp_order):
        node = c_comp_order.pop(0)
        for cond in c_component_and_tail:
            if topo_order.index(cond) < topo_order.index(node):
                if cond not in cond_vars:
                    cond_vars.append(cond)
                if cond not in Wc:
                    Wc.append(cond)
        symbolical_constraints_probabilities.append({node: cond_vars.copy()})
        cond_vars.clear()
    return symbolical_constraints_probabilities, Wc


def calculate_decision_matrix(
    mechanisms: MechanismType,
    Wc: list[Node],
    considered_c_comp: list[Node],
    unob: Node,
) -> list[list[int]]:
    """
    Calculates the decision matrix for the linear program, representing the coefficients for each mechanism.

    Args:
        mechanisms (MechanismType): Mechanism mappings for the model.
        Wc (list[Node]): Variables present in the constraints.
        considered_c_comp (list[Node]): Nodes in the considered c-component.
        unob (Node): The unobservable (latent) parent node from the intervention.

    Returns:
        list[list[int]]: The decision matrix as a list of lists of coefficients.
    """
    decision_matrix: list[list[int]] = [[1 for _ in range(len(mechanisms))]]
    spaces: list[list[int]] = [range(var.cardinality) for var in Wc]
    cartesian_product: list[list[int]] = MechanismGenerator.generate_cross_products(
        listSpaces=spaces
    )
    for realization in cartesian_product:
        aux: list[int] = []
        for u in range(len(mechanisms)):
            coef: bool = True
            for var in Wc:
                if var in considered_c_comp:
                    endo_parents: list[Node] = var.parents.copy()
                    endo_parents.remove(unob)
                    key = create_realization_string(
                        parents=endo_parents, realization=realization, indexer_list=Wc
                    )
                    endo_parents.clear()
                    if mechanisms[u][key] == realization[Wc.index(var)]:
                        coef *= 1
                    else:
                        coef *= 0
                        break
            aux.append(float(coef))
        decision_matrix.append(aux)
    return decision_matrix


def calculate_constraints_empirical_probabilities(
    data: pd.DataFrame,
    Wc: list[Node],
    symbolical_constraints_probabilities: list[dict[Node, list[Node]]],
) -> list[float]:
    """
    Calculates the empirical probabilities for each constraint in the linear program.

    Args:
        data (pd.DataFrame): The dataset containing observed variable values.
        Wc (list[Node]): Variables present in the constraints.
        symbolical_constraints_probabilities (list[dict[Node, list[Node]]]):
            List of dictionaries mapping each node to its conditioning variables.

    Returns:
        list[float]: List of empirical probabilities for each constraint.
    """
    probs: list[float] = [1.0]
    spaces: list[list[int]] = [range(var.cardinality) for var in Wc]
    cartesian_product: list[list[int]] = MechanismGenerator.generate_cross_products(
        listSpaces=spaces
    )
    for realization in cartesian_product:
        prob = 1.0
        for term in symbolical_constraints_probabilities:
            target_realization_nodes: list[Node] = []
            condition_realization_nodes: list[Node] = []
            for key_node in term:
                key_node.value = realization[Wc.index(key_node)]
                target_realization_nodes.append(key_node)
                for cVar in term[key_node]:
                    cVar.value = realization[Wc.index(cVar)]
                    condition_realization_nodes.append(cVar)
            prob *= find_conditional_probability(
                dataFrame=data,
                target_realization=target_realization_nodes,
                condition_realization=condition_realization_nodes,
            )
            target_realization_nodes.clear()
            condition_realization_nodes.clear()
        probs.append(prob)
    return probs
