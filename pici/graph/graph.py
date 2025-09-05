from __future__ import annotations

import networkx as nx

from pici.graph.node import Node


class Graph:
    def __init__(
        self,
        numberOfNodes: int,
        current_nodes: list[Node],
        dag_components: list[list[Node]],
        exogenous: list[Node],
        endogenous: list[Node],
        topological_order: list[Node],
        DAG: nx.DiGraph,
        c_component_to_unob: dict[int, Node],
        graph_nodes: dict[str, Node],
        node_set: set[Node],
        topological_order_indexes: dict[Node, int],
    ):
        self.number_of_nodes = numberOfNodes
        self.graph_nodes = graph_nodes
        self.curr_nodes = current_nodes
        self.dag_components = dag_components
        self.endogenous = endogenous
        self.exogenous = exogenous
        self.DAG = DAG
        self.c_component_to_unob = c_component_to_unob
        self.node_set = node_set
        self.topological_order = topological_order
        self.topological_order_indexes = topological_order_indexes

    def find_ancestors(self, target_node: Node) -> list[Node]:
        self.curr_nodes.clear()
        self._clear_visited()
        self._dfs_ancestor(target_node)
        ancestors: list[Node] = []
        for node in self.graph_nodes.values():
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
        return node_label in self.graph_nodes

    def set_node_intervened_value(self, node_label: str, node_value: int) -> Node:
        if not isinstance(node_value, int):
            raise Exception(f"Node value '{node_value}' is not of type int.")
        self.graph_nodes[node_label].intervened_value = node_value
        return self.graph_nodes[node_label]

    def get_closest_node_from_leaf_in_the_topological_order(
        self, nodes: list[Node]
    ) -> Node:
        n = len(self.topological_order) - 1
        for j in range(n, -1, -1):
            if self.topological_order[j] in nodes:
                return self.topological_order[j]
        raise Exception("Node not found")

    def _clear_visited(self):
        for node in self.node_set:
            node.visited = False

    def _base_dfs(self, node: Node):
        node.visited = True
        for adj_node in node.children:
            if not adj_node.visited:
                self._base_dfs(adj_node)

    def _dfs_ancestor(self, node: Node):
        node.visited = True

        for parent in node.parents:
            if not parent.visited:
                self._dfs_ancestor(parent)
