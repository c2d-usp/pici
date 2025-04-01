import networkx as nx
from pandas import DataFrame
import matplotlib.pyplot as plt

from causal_reasoning.graph.graph import Graph
from causal_reasoning.graph.node import Node
from causal_reasoning.utils.parser import (
    parse_to_int,
    parse_to_int_list,
    parse_to_string,
    parse_to_string_list,
    parse_default_input,
)
from causal_reasoning.linear_algorithm.opt_problem_builder import OptProblemBuilder


class CausalModel:
    def __init__(
        self,
        data: DataFrame,
        edges: str,
        unobservables: list[str] | str | None = [],
        interventions: str | list[str] | None = [],
        interventions_value: int | list[int] | None = [],
        target: str | None = "",
        target_value: int | None = None,
    ) -> None:
        self.data = data
        
        self.unobservables = parse_to_string_list(unobservables)

        self.interventions = parse_to_string_list(interventions)
        self.interventions_value = parse_to_int_list(interventions_value)

        self.target = target
        self.target_value = target_value


        # TODO: Adicionar interventions no objeto Graph.
        # Quando for fazer a query, usar os valores dados,
        # Caso não tenha dado, dar erro e falar que não foram
        # dadas as intervenções e o target
        self.graph = get_graph(str_graph=edges, unobservables=self.unobservables)

    def visualize_graph(self):
        """
        Create an image and plot the DAG
        """
        # TODO: Implement in the class Graph
        raise NotImplementedError

    # TODO: Re-think about how to add interventions with intervention values (the same to target)

    def _update_list(
        self, attr_name: str, values: list[str], reset: bool = False
    ) -> None:
        attr = getattr(self, attr_name)
        if reset:
            attr.clear()
        if self.is_nodes_in_graph(values):
            attr.extend(values)
            return
        raise Exception(f"Nodes '{values}' not present in the defined graph.")

    def add_interventions(self, interventions: list[str]) -> None:
        self._update_list("interventions", interventions)

    def set_interventions(self, interventions: list[str]) -> None:
        self._update_list("interventions", interventions, reset=True)

    def add_unobservables(self, unobservables):
        self._update_list("unobservables", unobservables)

    def set_unobservables(self, unobservables):
        self._update_list("unobservables", unobservables, reset=True)

    def set_target(self, target: str) -> None:
        self.target.clear()
        if self.is_nodes_in_graph([target]):
            self.target = target
            return
        raise Exception(f"Nodes '{target}' not present in the defined graph.")

    def is_nodes_in_graph(self, nodes: list[str]):
        for node in nodes:
            if node not in self.graph:
                return False
        return True

    def inference_query(
        self,
        interventions: list[str] | str | None = [],
        interventions_value: list[int] | int | None = [],
        target: str | None = "",
        target_value: int | None = None,
    ):
        if ( (interventions is None and self.interventions is None) \
            or (interventions_value is None and self.interventions_value is None) \
            or (target is None and self.target is None) \
            or (target_value is None and self.target_value is None)
        ):
            # TODO: Rewrite the error message
            raise Exception("Expect some value")
            
        # TODO: Set interventions and target to CausalModel object
        OptProblemBuilder.builder_linear_problem(
            self.graph,
            self.data,
            self.interventions,
            self.interventions_value,
            self.target,
            self.target_value,
        )

    def are_d_separated(
        self,
        set_nodes_X: list[str],
        set_nodes_Y: list[str],
        set_nodes_Z: list[str],
    ) -> bool:
        '''
            Is set of nodes X d-separated from set of nodes Y through set of nodes Z?
        '''
        # TODO: Usando nx é muito mais fácil,
        # porém não tem o conceito de não observável.
        # É possível atribuir atributos aos nós
        # nx.set_node_attributes(G, {3: "unobservable", 4: "unobservable"}, "status")

        
        # plt.figure(figsize=(6, 6))
        # pos = nx.spring_layout(G)
        # nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, arrowsize=20)

        # # Save the image
        # plt.savefig("digraph.png", dpi=300, bbox_inches='tight')
        # plt.show()
        a = nx.is_d_separator(G, set_nodes_X, set_nodes_Y, set_nodes_Z)
        print(f"A:::{a}")
        return self.graph.check_dseparation(set_nodes_X, set_nodes_Y, set_nodes_Z)


def get_graph(str_graph: str = None, unobservables: list[str] = None):
    auxTuple = parse_default_input(str_graph, latents=unobservables)

    numberOfNodes, labelToIndex, indexToLabel, adj, cardinalities, parents = auxTuple

    inpDAG: nx.DiGraph = nx.DiGraph()
    for i in range(numberOfNodes):
        inpDAG.add_node(i)

    for parent, edge in enumerate(adj):
        if bool(edge):
            for ch in edge:
                inpDAG.add_edge(parent, ch)
    order = list(nx.topological_sort(inpDAG))

    for i in range(numberOfNodes):
        name_node = indexToLabel[i]
        nx.relabel_nodes(inpDAG, {i: name_node}, copy=False)

    endogenIndex: list[int] = []
    exogenIndex: list[int] = []
    for i in range(numberOfNodes):
        if not (bool(parents[i])):
            exogenIndex.append(i)
        else:
            endogenIndex.append(i)

    graphNodes: list[Node] = [
        Node(latentParent=-1, parents=[], children=[], isLatent=False)
        for _ in range(numberOfNodes)
    ]
    for node in range(numberOfNodes):
        if cardinalities[node] == 0:
            graphNodes[node] = Node(
                children=adj[node], parents=[], latentParent=None, isLatent=True
            )
        else:
            latentParent = -1
            for nodeParent in parents[node]:
                if cardinalities[nodeParent] == 0:
                    latentParent = nodeParent
                    break

            if latentParent == -1:
                print(
                    f"PARSE ERROR: ALL OBSERVABLE VARIABLES SHOULD HAVE A LATENT PARENT, BUT {node} DOES NOT."
                )

            graphNodes[node] = Node(
                children=adj[node],
                parents=parents[node],
                latentParent=latentParent,
                isLatent=False,
            )
        pass

    return Graph(
        numberOfNodes=numberOfNodes,
        currNodes=[],
        visited=[False] * (numberOfNodes),
        cardinalities=cardinalities,
        parents=parents,
        adj=adj,
        indexToLabel=indexToLabel,
        labelToIndex=labelToIndex,
        dagComponents=[],
        exogenous=exogenIndex,
        endogenous=endogenIndex,
        topologicalOrder=order,
        DAG=inpDAG,
        cComponentToUnob={},
        graphNodes=graphNodes,
        moralGraphNodes=[],
    )


def parse_str_to_nx_graph(edges_str: str, latents: list[str]) -> str:
    custom_cardinalities = {}

    edges_part = edges_str.split(",")
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
            node_card[node] = 0 if node in latents else 2

    u_nodes = [n for n in node_order if n in latents]
    other_nodes = [n for n in node_order if not n in latents]
    final_node_order = u_nodes + other_nodes

    lines = []

    lines.append(str(len(final_node_order)))  # no nodes
    lines.append(str(len(edges)))  # no edges

    for node in final_node_order:
        lines.append(f"{node} {node_card[node]}")

    for left, right in edges:
        lines.append(f"{left} {right}")

    return "\n".join(lines)
