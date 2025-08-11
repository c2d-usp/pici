from __future__ import annotations

import networkx as nx

from pici.graph.node import Node


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
        graphNodes: dict[str, Node],
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

    def find_ancestors(self, target_node: Node) -> list[Node]:
        self.currNodes.clear()
        self._clear_visited()
        self._dfs_ancestor(target_node)
        ancestors: list[Node] = []
        for node in self.graphNodes.values():
            if node.visited:
                ancestors.append(node)
        return ancestors

    def is_descendant(self, ancestor: Node, descendant: Node) -> bool:
        if ancestor not in self.node_set or descendant not in self.node_set:
            return True
        self._clear_visited()
        self._base_dfs(node=ancestor)
        return descendant.visited

    def is_node_in_graph(self, node_label: str) -> bool:
        if not isinstance(node_label, str):
            raise Exception(f"Node label '{node_label}' is not of type {str}.")
        return node_label in self.graphNodes

    def set_node_intervened_value(self, node_label: str, node_value: int) -> Node:
        if not isinstance(node_value, int):
            raise Exception(f"Node value '{node_value}' is not of type int.")
        # TODO: Validate if value fits in node cardinality
        self.graphNodes[node_label].intervened_value = node_value
        return self.graphNodes[node_label]

    def get_closest_node_from_leaf_in_the_topological_order(
        self, nodes: list[Node]
    ) -> Node:
        n = len(self.topologicalOrder) - 1
        for j in range(n, -1, -1):
            if self.topologicalOrder[j] in nodes:
                return self.topologicalOrder[j]

        # TODO: BETTER ERROR HANDLING
        raise Exception("Node not found")

    # Not Used
    def find_cComponents(self):
        for node in self.node_set:
            if not node.visited and node.cardinality < 1:
                self.currNodes.clear()
                self._visit_nodes_in_same_cComponent(node)
                self.dagComponents.append(self.currNodes[:])
                self.cComponentToUnob[len(self.dagComponents) - 1] = node

    def _clear_visited(self):
        for node in self.node_set:
            node.visited = False

    def _base_dfs(self, node: Node):
        node.visited = True
        for adj_node in node.children:
            if not adj_node.visited:
                self._base_dfs(adj_node)

    # Only used by 'find_cComponents'
    def _visit_nodes_in_same_cComponent(self, node: Node):
        node.visited = True
        self.currNodes.append(node)
        is_observable = node.cardinality > 1

        if not is_observable:
            for adj_node in node.children:
                if not adj_node.visited:
                    self._visit_nodes_in_same_cComponent(adj_node)
        else:
            for parent_node in node.parents:
                if not parent_node.visited and parent_node.cardinality < 1:
                    self._visit_nodes_in_same_cComponent(parent_node)

    def _dfs_ancestor(self, node: Node):
        node.visited = True

        for parent in node.parents:
            if not parent.visited:
                self._dfs_ancestor(parent)
