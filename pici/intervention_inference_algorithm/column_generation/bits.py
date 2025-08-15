
# Mas e se tiver duas Latentes?
def generate_latent_bit_indices(latent) -> list[int]:
    """
    Creates a list of bit indices representing the latent variable's configuration.

    Args:
        latent: The latent variable .

    Returns:
        list[int]: A list of bit indices for the latent variable's configuration.
    """
    n = calculate_latent_bit_length(latent=latent)
    bits = [i for i in range(n)]
    return bits

def calculate_latent_bit_length(latent) -> int:
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
            if parent.is_latent:
                continue
            child_bit_length *= parent.cardinality
        latent_bit_length += child_bit_length
    return latent_bit_length
