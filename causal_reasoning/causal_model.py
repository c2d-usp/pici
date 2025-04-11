import networkx as nx
from pandas import DataFrame

from causal_reasoning.graph.graph import Graph
from causal_reasoning.graph.node import Node, T
from causal_reasoning.linear_algorithm.opt_problem_builder import \
    builder_linear_problem
from causal_reasoning.utils.parser import (parse_input_graph,
                                           parse_to_string_list)


class CausalModel:
    def __init__(
        self,
        data: DataFrame,
        edges: str,
        unobservables_labels: list[T] | T | None = [],
        interventions: list[tuple[T, int]] = [],
        target: tuple[T, int] = None,
    ) -> None:
        self.data = data
        unobservables_labels = parse_to_string_list(unobservables_labels)

        self.graph: Graph = get_graph(edges=edges, unobservables=unobservables_labels)
        self.unobservables = [
            self.graph.graphNodes[unobservable_label]
            for unobservable_label in unobservables_labels
        ]
        self.interventions = self.parse_list_tuples_into_list_nodes(interventions)
        self.target = self.parse_tuple_into_node(target)

    def parse_list_tuples_into_list_nodes(
        self, list_tuples_label_value: list[tuple[T, int]]
    ):
        if list_tuples_label_value is None or len(list_tuples_label_value) <= 0:
            return None
        return [self.parse_tuple_into_node(tuple) for tuple in list_tuples_label_value]

    def parse_tuple_into_node(self, tuple_label_value: tuple[T, int]):
        if tuple_label_value is None:
            return None
        label, value = tuple_label_value
        if not self.is_node_in_graph(label):
            raise Exception(f"Node '{label}' not present in the defined graph.")
        self.set_node_value(label, value)
        return self.graph.graphNodes[label]

    def is_node_in_graph(self, node_label: T) -> bool:
        return node_label in self.graph.graphNodes

    def set_node_value(self, node_label: T, node_value: int) -> Node:
        # TODO: Validate if value fits in node cardinality
        self.graph.graphNodes[node_label].value = node_value
        return self.graph.graphNodes[node_label]

    def set_interventions(self, interventions: list[tuple[T, int]]) -> None:
        self.interventions = self.parse_list_tuples_into_list_nodes(interventions)

    def add_interventions(self, interventions: list[tuple[T, int]]) -> None:
        more_interventions = self.parse_list_tuples_into_list_nodes(interventions)
        if more_interventions is None:
            return
        for intervention in more_interventions:
            if intervention not in self.interventions:
                self.interventions.append(intervention)

    def set_target(self, target: tuple[T, int]) -> None:
        self.target = self.parse_tuple_into_node(target)

    def single_intervention_query(self):
        builder_linear_problem(
            self.graph,
            self.data,
            self.interventions[0],
            self.target,
        )

    def multi_intervention_query(self):
        raise NotImplementedError

    def visualize_graph(self):
        raise NotImplementedError

    def add_unobservables(self, unobservables):
        # This implies the whole graph re-creation
        # Changes the intervention and target also (?)
        raise NotImplementedError

    def set_unobservables(self, unobservables):
        # This implies the whole graph re-creation
        # Changes the intervention and target also (?)
        raise NotImplementedError

    def inference_query(
        self, interventions: list[tuple[T, int]] = [], target: tuple[T, int] = None
    ):
        interventions_nodes = self.parse_list_tuples_into_list_nodes(interventions)
        if interventions_nodes is None and self.interventions is None:
            raise Exception("Expect intervention to be not None")

        if interventions_nodes is not None:
            self.interventions = interventions_nodes

        target_node = self.parse_tuple_into_node(target)
        if target_node is None and self.target is None:
            raise Exception("Expect target to be not None")
        if target_node is not None:
            self.target = target_node

        if len(self.interventions) > 1:
            self.multi_intervention_query()
            return
        self.single_intervention_query()

    def are_d_separated(
        self,
        set_nodes_X: list[str],
        set_nodes_Y: list[str],
        set_nodes_Z: list[str],
    ) -> bool:
        """
        Is set of nodes X d-separated from set of nodes Y through set of nodes Z?
        """
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
        # a = nx.is_d_separator(G, set_nodes_X, set_nodes_Y, set_nodes_Z)
        # print(f"A:::{a}")
        return self.graph.check_dseparation(
            get_node_list(self.graph.graphNodes, set_nodes_X),
            get_node_list(self.graph.graphNodes, set_nodes_Y),
            get_node_list(self.graph.graphNodes, set_nodes_Z),
        )


def get_node(graphNodes: dict[T, Node], node_label: T):
    return graphNodes[node_label]


def get_node_list(graphNodes: dict[T, Node], node_labels: list[T]) -> list[Node]:
    return [get_node(graphNodes, node_label) for node_label in node_labels]


def get_parent_latent(parents_label: list[T], node_cardinalities: list[T]) -> T:
    for node_parent in parents_label:
        if node_cardinalities[node_parent] == 0:
            return node_parent
    return None


def get_graph(
    edges: T = None,
    unobservables: list[T] = None,
    custom_cardinalities: dict[T, int] = None,
):
    (
        number_of_nodes,
        children_labels,
        node_cardinalities,
        parents_labels,
        node_labels_set,
        dag,
    ) = parse_input_graph(edges, latents_label=unobservables)
    order = list(nx.topological_sort(dag))

    parent_latent_labels: dict[T, T] = {}
    graphNodes: dict[T, Node] = {}
    node_set: set[Node] = set()

    parent_latent_label: T = None
    for node_label in node_labels_set:
        if node_cardinalities[node_label] == 0:
            parent_latent_label = None
            new_node = Node(
                children=[],
                parents=[],
                latentParent=None,
                isLatent=True,
                label=node_label,
                cardinality=node_cardinalities[node_label],
            )
        else:
            parent_latent_label = get_parent_latent(
                parents_labels[node_label], node_cardinalities
            )

            if parent_latent_label == None:
                # TODO: ADD WARNING MESSAGE
                print(
                    f"PARSE ERROR: ALL OBSERVABLE VARIABLES SHOULD HAVE A LATENT PARENT, BUT {node_label} DOES NOT."
                )

            new_node = Node(
                children=[],
                parents=[],
                latentParent=None,
                isLatent=False,
                label=node_label,
                cardinality=node_cardinalities[node_label],
            )

        graphNodes[node_label] = new_node
        parent_latent_labels[new_node.label] = parent_latent_label
        node_set.add(new_node)

    endogenous: list[Node] = []
    exogenous: list[Node] = []
    topologicalOrderIndexes = {}

    for i, node_label in enumerate(node_labels_set):
        node = graphNodes[node_label]
        if node.isLatent:
            exogenous.append(node)
            node.children = get_node_list(graphNodes, children_labels[node.label])
        else:
            node.latentParent = graphNodes[parent_latent_labels[node_label]]
            endogenous.append(node)
            node.children = get_node_list(graphNodes, children_labels[node.label])
            node.parents = get_node_list(graphNodes, parents_labels[node.label])
        topologicalOrderIndexes[node] = i

    topological_order_nodes: list[Node] = []
    for node_label in order:
        topological_order_nodes.append(graphNodes[node_label])

    return Graph(
        numberOfNodes=number_of_nodes,
        exogenous=exogenous,  # list[Node]
        endogenous=endogenous,  # list[Node]
        topologicalOrder=topological_order_nodes,  # list[Node]
        DAG=dag,  # nx.DiGraph
        graphNodes=graphNodes,  # dict[str, Node]
        node_set=node_set,  # set(Node)
        topologicalOrderIndexes=topologicalOrderIndexes,  # dict[Node, int]
        currNodes=[],
        dagComponents=[],  # list[list[Node]]
        cComponentToUnob={},  # dict[int, Node]
    )
