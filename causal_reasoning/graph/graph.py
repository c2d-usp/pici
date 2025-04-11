import networkx as nx

from causal_reasoning.graph.node import Node, T


class Graph:
    def __init__(
        self,
        numberOfNodes: int,
        currNodes: list[Node],
        dagComponents: list[list[Node]],
        exogenous: list[Node],
        endogenous: list[Node],
        topologicalOrder: list[Node],
        DAG: nx.DiGraph,
        cComponentToUnob: dict[int, Node],
        graphNodes: dict[T, Node],
        node_set: set[Node],
        topologicalOrderIndexes: dict[Node, int],
    ):
        self.numberOfNodes = numberOfNodes
        self.graphNodes = graphNodes
        self.currNodes = currNodes
        self.dagComponents = dagComponents
        self.endogenous = endogenous
        self.exogenous = exogenous
        self.topologicalOrder = topologicalOrder
        self.DAG = DAG
        self.cComponentToUnob = cComponentToUnob
        self.node_set = node_set
        self.topologicalOrderIndexes = topologicalOrderIndexes

    def visit_nodes_in_same_cComponent(self, node: Node):
        node.visited = True
        self.currNodes.append(node)
        is_observable = node.cardinality > 1

        if not is_observable:
            for adj_node in node.children:
                if not adj_node.visited:
                    self.visit_nodes_in_same_cComponent(adj_node)
        else:
            for parent_node in node.parents:
                if not parent_node.visited and parent_node.cardinality < 1:
                    self.visit_nodes_in_same_cComponent(parent_node)

    def find_cComponents(self):
        for node in self.node_set:
            if not node.visited and node.cardinality < 1:
                self.currNodes.clear()
                self.visit_nodes_in_same_cComponent(node)
                self.dagComponents.append(self.currNodes[:])
                self.cComponentToUnob[len(self.dagComponents) - 1] = node

    def base_dfs(self, node: Node):
        node.visited = True
        for adj_node in node.children:
            if not adj_node.visited:
                self.base_dfs(adj_node)

    def is_descendant(self, ancestor: Node, descendant: Node) -> bool:
        if ancestor not in self.node_set or descendant not in self.node_set:
            return True
        self.clear_visited()
        self.base_dfs(node=ancestor)
        return descendant.visited

    def clear_visited(self):
        for node in self.node_set:
            node.visited = False

    def get_closest_node_from_leaf_in_the_topological_order(
        self, nodes: list[Node]
    ) -> Node:
        higher_idx = 0
        higher_node = ""
        for node in nodes:
            idx = self.topologicalOrderIndexes[node]
            if idx >= higher_idx:
                higher_idx = idx
                higher_node = node
        return higher_node

    def build_moral(
        self,
        consideredNodes: list[Node],
        conditionedNodes: list[Node],
        intervention_outgoing_edges_are_considered=True,
        intervention: Node = None,
    ):
        """
        Builds the moral graph, considering only part of the nodes.
        intervention_outgoing_edges_are_considered: if true, the outgoing edges of the intervention should be considered.
        are_intervention_outgoing_edges_considered of the intervention
        """
        for node in self.node_set:
            node.moral_adjacency = []

            if node not in consideredNodes:
                continue

            if node in conditionedNodes:
                for parent1 in node.parents:
                    if (
                        not intervention_outgoing_edges_are_considered
                        and parent1 == intervention
                    ):
                        continue
                    for parent2 in node.parents:
                        if (
                            not intervention_outgoing_edges_are_considered
                            and parent2 == intervention
                        ):
                            continue

                        if parent1 in conditionedNodes and parent2 in consideredNodes:
                            if parent2 not in parent1.moral_adjacency:
                                parent1.moral_adjacency.append(parent2)
                            if parent1 not in parent2.moral_adjacency:
                                parent2.moral_adjacency.append(parent1)
            else:
                if (
                    not intervention_outgoing_edges_are_considered
                    and node == intervention
                ):
                    continue

                for child in node.children:
                    if child in consideredNodes and child not in conditionedNodes:
                        if node not in child.moral_adjacency:
                            child.moral_adjacency.append(node)
                        if child not in node.moral_adjacency:
                            node.moral_adjacency.append(child)

    def find_ancestors(self, target_node: Node) -> list[Node]:
        self.currNodes.clear()
        self.clear_visited()
        self.dfs_ancestor(target_node)
        ancestors: list[Node] = []
        for node in self.graphNodes.values():
            if node.visited:
                ancestors.append(node)
        return ancestors

    def dfs_ancestor(self, node: Node):
        node.visited = True

        for parent in node.parents:
            if not parent.visited:
                self.dfs_ancestor(parent)

    def independency_moral(self, node_1: Node, node_2: Node) -> bool:
        self.clear_visited()
        self.dfs_moral(node_1)

        return not node_2.visited

    def dfs_moral(self, node: Node):
        node.visited = True

        for adj in node.moral_adjacency:
            if not adj.visited:
                self.dfs_moral(adj)

    def check_dseparation(
        self,
        set_nodes_1: list[Node],
        set_nodes_2: list[Node],
        conditioned_nodes: list[Node],
    ) -> bool:
        """
        Given two sets of nodes (nodes1 and nodes2), the function returns true if every node in nodes1
        is independent of every node in nodes2, given that the nodes in conditionedNodes are conditioned.
        """

        self.build_moral(
            consideredNodes=list(self.node_set),
            conditionedNodes=conditioned_nodes,
        )

        self.clear_visited()
        for node in set_nodes_1:
            if not node.visited:
                self.dfs_moral(node)

        areDseparated = True
        for node in set_nodes_2:
            if node.visited:
                areDseparated = False
                break
        return areDseparated
