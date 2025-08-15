from pici.graph.node import Node


def generate_optimization_problem_bit_list(intervention: Node) -> list[int]:
    """
    Creates a list of bits. Each bit representing the value assumed by a variable in response to a realization of its parents.

    Args:
        intervention: The intervention variable.

    Returns:
        list[int]: A list of the bit indices for the problem optimization variable's configuration.
    """
    n = calculate_latent_bit_length(latent=intervention.latentParent)
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
        child_bit_length = 0
        for parent in child.parents:
            if parent.isLatent:
                continue
            child_bit_length *= parent.cardinality
        latent_bit_length += child_bit_length
    return latent_bit_length
