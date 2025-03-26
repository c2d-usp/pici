import networkx as nx
from causal_reasoning.graph.moral_node import MoralNode
from causal_reasoning.graph.node import Node


class Graph:
    def __init__(
        self,
        numberOfNodes: int,
        currNodes: list[int],
        visited: list[bool],
        cardinalities: dict[int, int],
        parents: list[list[int]],
        adj: list[list[int]],
        labelToIndex: dict[str, int],
        indexToLabel: dict[int, str],
        dagComponents: list[list[int]],
        exogenous: list[int],
        endogenous: list[int],
        topologicalOrder: list[int],
        DAG: nx.digraph,
        cComponentToUnob: dict[int, int],
        graphNodes: list[Node],
        moralGraphNodes: list[MoralNode],
    ):
        self.numberOfNodes = numberOfNodes
        self.currNodes = currNodes
        self.visited = visited
        self.cardinalities = cardinalities
        self.parents = parents
        self.adj = adj
        self.labelToIndex = labelToIndex
        self.indexToLabel = indexToLabel
        self.dagComponents = dagComponents
        self.endogenous = endogenous
        self.exogenous = exogenous
        self.topologicalOrder = topologicalOrder
        self.DAG = DAG
        self.cComponentToUnob = cComponentToUnob
        self.graphNodes = graphNodes
        self.moralGraphNodes = moralGraphNodes

    def dfs(self, node: int):
        self.visited[node] = True
        self.currNodes.append(node)
        is_observable = self.cardinalities[node] > 1

        if not is_observable:
            for adj_node in self.adj[node]:
                if not self.visited[adj_node]:
                    self.dfs(adj_node)
        else:
            for parent_node in self.parents[node]:
                if (
                    not self.visited[parent_node]
                    and self.cardinalities[parent_node] < 1
                ):
                    self.dfs(parent_node)

    def find_cComponents(self):
        for i in range(self.numberOfNodes):
            if not self.visited[i] and self.cardinalities[i] < 1:
                self.currNodes.clear()
                self.dfs(i)
                self.dagComponents.append(self.currNodes[:])
                self.cComponentToUnob[len(self.dagComponents) - 1] = i

    def base_dfs(self, node: int):
        self.visited[node] = True
        for adj_node in self.graphNodes[node].children:
            if not self.visited[adj_node]:
                self.base_dfs(adj_node)

    def is_descendant(self, ancestor, descendant):
        for i in range(len(self.visited)):
            self.visited[i] = False
        self.base_dfs(node=ancestor)
        return self.visited[descendant]

    def build_moral(
        self,
        consideredNodes: list[int],
        conditionedNodes: list[int],
        flag=False,
        intervention=-1,
    ):
        """
        Builds the moral graph, considering only part of the nodes.
        flag: if true, the outgoing edges of the intervention should not be considered.
        """
        self.moralGraphNodes = [
            MoralNode(adjacent=[]) for _ in range(self.numberOfNodes)
        ]
        for node in range(self.numberOfNodes):
            if node not in consideredNodes:
                continue

            if node in conditionedNodes:
                for parent1 in self.graphNodes[node].parents:
                    if flag and parent1 == intervention:
                        continue
                    for parent2 in self.graphNodes[node].parents:
                        if flag and parent2 == intervention:
                            continue

                        if parent1 in conditionedNodes and parent2 in consideredNodes:
                            if parent2 not in self.moralGraphNodes[parent1].adjacent:
                                self.moralGraphNodes[parent1].adjacent.append(parent2)
                            if parent1 not in self.moralGraphNodes[parent2].adjacent:
                                self.moralGraphNodes[parent2].adjacent.append(parent1)
            else:
                if flag and node == intervention:
                    continue

                for ch in self.graphNodes[node].children:
                    if ch in consideredNodes and ch not in conditionedNodes:
                        if node not in self.moralGraphNodes[ch].adjacent:
                            self.moralGraphNodes[ch].adjacent.append(node)
                        if ch not in self.moralGraphNodes[node].adjacent:
                            self.moralGraphNodes[node].adjacent.append(ch)

    def find_ancestors(self, node: int):
        self.currNodes.clear()
        self.visited = [False] * self.numberOfNodes
        self.dfs_ancestor(node)
        ancestors: list[int] = []
        for i in range(0, self.numberOfNodes):
            if self.visited[i]:
                ancestors.append(i)
        return ancestors

    def dfs_ancestor(self, node):
        self.visited[node] = True

        for parent in self.graphNodes[node].parents:
            if not self.visited[parent]:
                self.dfs_ancestor(parent)

    def independency_moral(self, node1: int, node2: int):
        self.visited = [False] * self.numberOfNodes
        self.dfs_moral(node1)

        return not self.visited[node2]

    def dfs_moral(self, node):
        self.visited[node] = True

        for adj in self.moralGraphNodes[node].adjacent:
            if not self.visited[adj]:
                self.dfs_moral(node=adj)
