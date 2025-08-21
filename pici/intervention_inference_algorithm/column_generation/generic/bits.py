from pici.graph.node import Node


def generate_optimization_problem_bit_list(intervention: Node) -> list[int]:
    """
    Creates a list of bits. Each bit representing the value assumed by a variable in response to a realization of its parents.

    Args:
        intervention: The intervention variable.

    Returns:
        list[int]: A list of the bit indices for the problem optimization variable's configuration.
    """
    n = calculate_latent_bit_length(latent=intervention.latent_parent)
    bits = [i for i in range(n)]
    return bits


def calculate_latent_bit_length(latent: Node) -> int:
    """
    Calculates the total number of bits required to represent the latent variable's configuration
    based on its endogenous children and their endogenous parents' cardinalities.

    For each child of the latent variable, this function multiplies the cardinalities of all
    endogenous (non-latent) parents and sums these products across all children.

    Args:
        latent: The latent variable node.

    Returns:
        int: The total number of bits required.
    """
    latent_bit_length = 0
    for child in latent.children:
        latent_bit_length += count_endogenous_parent_configurations(child)
    return latent_bit_length


def count_endogenous_parent_configurations(node: Node):
    """
    Calculates the number of unique configurations for a node's endogenous (non-latent) parents.

    For the given node, this function multiplies the cardinalities of all its non-latent parents,
    representing the number of possible value combinations those parents can take.

    Args:
        node (Node): The node whose endogenous parent configurations are counted.

    Returns:
        int: The total number of unique configurations for the node's endogenous parents.
    """
    node_bit_length = 1
    for parent in node.parents:
        if parent.is_latent:
            continue
        node_bit_length *= parent.cardinality    
    return node_bit_length

