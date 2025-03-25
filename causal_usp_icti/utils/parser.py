import networkx as nx


def parse_state(state):
    if isinstance(state, str):
        return [state]
    if isinstance(state, list):
        return state
    raise Exception(f"Input format for {state} not recognized: {type(state)}")


def parse_target(state):
    if isinstance(state, str):
        return state
    raise Exception(f"Input format for {state} not recognized: {type(state)}")


def parse_edges(state):
    if isinstance(state, str):
        # TODO: Verify if it is a valid string
        # TODO: Parse the string into nx.Graph
        return ""
    if isinstance(state, nx.Graph):
        # TODO: Verify if it is a valid string
        # TODO: Parse the string into nx.Graph
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
    versao_str: str, latent: list[str]) -> str:
    """
    Converts a string of edges like:
      "U1 -> X, U1 -> Y, U2 -> Z, X -> Y, Z -> X"
    into the a string that follows the pattern:
      5 # Number of nodes \n
      5 # Number of egdes \n
      U1 0 # Node U1 has cardinality 0 \n
      U2 0 # Node U2 has cardinality 0 \n
      X 2 # Node X has cardinality 2 \n
      Y 2 # Node Y has cardinality 2 \n
      Z 2 # Node Z has cardinality 2 \n
      U1 X # Edge U1 -> X \n
      U1 Y # Edge U1 -> Y \n
      U2 Z # Edge U2 -> Z \n
      X Y # Edge X -> Y \n
      Z X # Edge Z -> X
    """
    custom_cardinalities = {}

    edges_part = versao_str.split(",")
    edges = []
    node_order = []
    node_set = set()

    for part in edges_part:
        part = part.strip()
        left, right = part.split("->")
        left = left.strip()
        right = right.strip()

        edges.append((left, right))

        for n in (left, right):
            if n not in node_set:
                node_order.append(n)
                node_set.add(n)

    node_card = {}
    for node in node_order:
        if node in custom_cardinalities:
            node_card[node] = custom_cardinalities[node]
        else:
            node_card[node] = 0 if node in latent else 2

    u_nodes = [n for n in node_order if n in latent]
    other_nodes = [n for n in node_order if n not in latent]
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

    adj: list[list[int]] = [[] for _ in range(numberOfNodes)]
    parents: list[list[int]] = [[] for _ in range(numberOfNodes)]
    for left, right in edges:
        uIndex = labelToIndex[left]
        vIndex = labelToIndex[right]
        adj[uIndex].append(vIndex)
        parents[vIndex].append(uIndex)

    return numberOfNodes, labelToIndex, indexToLabel, adj, cardinalities, parents


def parse_file(file_path: str):
    with open(file_path, "r") as file:

        numberOfNodes = file.readline().strip()
        numberOfEdges = file.readline().strip()

        numberOfNodes = int(numberOfNodes)
        numberOfEdges = int(numberOfEdges)

        labelToIndex: dict[str, int] = {}
        indexToLabel: dict[int, str] = {}
        cardinalities: dict[int, int] = {}
        adj: list[list[int]] = [[] for _ in range(numberOfNodes)]
        parents: list[list[int]] = [[] for _ in range(numberOfNodes)]

        for i in range(numberOfNodes):
            label, cardinality = file.readline().strip().split()

            cardinality = int(cardinality)
            labelToIndex[label] = i
            indexToLabel[i] = label
            cardinalities[i] = cardinality

        for _ in range(numberOfEdges):
            u, v = file.readline().strip().split()
            uIndex = labelToIndex[u]
            vIndex = labelToIndex[v]
            adj[uIndex].append(vIndex)
            parents[vIndex].append(uIndex)

        return numberOfNodes, labelToIndex, indexToLabel, adj, cardinalities, parents


def parse_interface(nodesString: str, edgesString: str):
    nodesAndCardinalitiesList: list[str] = nodesString.split(",")
    numberOfNodes = len(nodesAndCardinalitiesList)

    cardinalities: dict[int, int] = {}
    labelToIndex: dict[str, int] = {}
    indexToLabel: dict[int, str] = {}
    adj: list[list[int]] = [[] for _ in range(numberOfNodes)]
    parents: list[list[int]] = [[] for _ in range(numberOfNodes)]

    for i, element in enumerate(nodesAndCardinalitiesList):
        auxPair = element.split("=")
        cardinalities[i] = auxPair[1]
        labelToIndex[auxPair[0]] = i
        indexToLabel[i] = auxPair[0]
        cardinalities[i] = int(auxPair[1])

    for element in edgesString.split(","):
        elAux = element.split("->")
        fatherIndex = labelToIndex[elAux[0]]
        sonIndex = labelToIndex[elAux[1]]
        adj[fatherIndex].append(sonIndex)
        parents[sonIndex].append(fatherIndex)

    return numberOfNodes, labelToIndex, indexToLabel, adj, cardinalities, parents
