import pandas as pd

from pici.graph.node import Node
from pici.utils.probabilities_helper import find_conditional_probability

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
    considered_c_comp_in_topo_order: list[Node], c_component_and_tail: list[Node], topo_order: list[Node]
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
    c_comp_order = considered_c_comp_in_topo_order.copy()
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
    
    if Wc is None:
        raise Exception("W is None")
    
    return symbolical_constraints_probabilities, Wc

def calculate_constraints_empirical_probabilities(
    data: pd.DataFrame,
    symbolical_constraints_probabilities: list[dict[Node, list[Node]]],
    reversed_ordered_W_realizations: list[list] = None,
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
    probs: list[float] = []
    header = reversed_ordered_W_realizations[0]
    cartesian_product = reversed_ordered_W_realizations[1:]
    for realization in cartesian_product:
        prob = 1.0
        for conditional_probability in symbolical_constraints_probabilities:
            target_realization_nodes: list[Node] = []
            condition_realization_nodes: list[Node] = []
            for target, conditioned_nodes in conditional_probability.items():
                target.value = realization[header.index(target.label)]
                target_realization_nodes.append(target)
                for cVar in conditioned_nodes:
                    cVar.value = realization[header.index(cVar.label)]
                    condition_realization_nodes.append(cVar)
            curr_prob = find_conditional_probability(
                dataFrame=data,
                target_realization=target_realization_nodes,
                condition_realization=condition_realization_nodes,
            )
            prob *= curr_prob
            target_realization_nodes.clear()
            condition_realization_nodes.clear()
        probs.append(prob)
    #probs.append(1)
    return probs

def calculate_number_of_constraints(W: list[Node]):
    """
    Receive W set
    """
    n_constraints = 1
    for node in W:
        n_constraints *= node.cardinality
    return n_constraints
    