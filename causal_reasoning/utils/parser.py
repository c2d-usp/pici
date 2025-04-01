import networkx as nx

# Create DAG with latents
# Validate if the DAG is a DAG




def parse_to_string_list(state):
    if isinstance(state, str):
        return [state]
    if isinstance(state, list):
        if all(isinstance(item, str) for item in state):
            return state
    raise Exception(f"Input format for {state} not recognized.")


def parse_to_string(state):
    if isinstance(state, str):
        return state
    raise Exception(f"Input format for {state} not recognized: {type(state)}")


def parse_to_int(state):
    if isinstance(state, int):
        return state
    raise Exception(f"Input format for {state} not recognized: {type(state)}")


def parse_to_int_list(state):
    if isinstance(state, int):
        return [state]
    if isinstance(state, list):
        if all(isinstance(item, int) for item in state):
            return state
    raise Exception(f"Input format for {state} not recognized.")


def parse_interventions(intervention_variables, intervention_values):
    if isinstance(intervention_variables, str) and isinstance(intervention_values, int):
        return 


def parse_edges(state):
    if isinstance(state, str):
        # TODO: Verify if it is a valid string
        # TODO: Parse the string into nx.Graph
        return ""
    if isinstance(state, nx.Graph):
        # TODO: Verify if it is a valid nx.Graph
        return ""
    if isinstance(state, list):
        # TODO: Verify if it is list of tuples
        # TODO: Parse the tuples into nx.Graph
        return ""
    if isinstance(state, tuple):
        # TODO: Parse the tuples into nx.Graph
        return ""
    raise Exception(f"Input format for {state} not recognized: {type(state)}")


def parse_default_input(
    edges: str, latents: list[str]) -> str:

    custom_cardinalities = {}

    edges_part = edges.split(",")
    edges_tuples = []
    node_order = []
    node_set = set()

    for part in edges_part:
        part = part.strip()
        left, right = part.split("->")
        left = left.strip()
        right = right.strip()

        edges_tuples.append((left, right))

        for n in (left, right):
            if n not in node_set:
                node_order.append(n)
                node_set.add(n)

    node_card = {}
    for node in node_order:
        if node in custom_cardinalities:
            node_card[node] = custom_cardinalities[node]
        else:
            node_card[node] = 0 if node in latents else 2

    u_nodes = [n for n in node_order if n in latents]
    other_nodes = [n for n in node_order if n not in latents]
    final_node_order = u_nodes + other_nodes

    numberOfNodes = len(final_node_order)

    labelToIndex: dict[str, int] = {}
    indexToLabel: dict[int, str] = {}
    cardinalities: dict[int, int] = {}

    for index, node in enumerate(final_node_order):
        cardinality = int(node_card[node])
        labelToIndex[node] = index
        indexToLabel[index] = node
        cardinalities[index] = cardinality

    adjacency_list: list[list[int]] = [[] for _ in range(numberOfNodes)]
    parents: list[list[int]] = [[] for _ in range(numberOfNodes)]
    for left, right in edges_tuples:
        uIndex = labelToIndex[left]
        vIndex = labelToIndex[right]
        adjacency_list[uIndex].append(vIndex)
        parents[vIndex].append(uIndex)

    return numberOfNodes, labelToIndex, indexToLabel, adjacency_list, cardinalities, parents
