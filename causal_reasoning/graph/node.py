from typing import Any
from typing import TypeVar, Generic

T = TypeVar("T")


class Node(Generic[T]):
    def __init__(self, children, parents, latentParent, isLatent, label: T):
        self.children: list[Node] = children
        self.parents: list[Node] = parents
        self.latentParent: Node = latentParent
        self.isLatent: bool = isLatent
        self.label: T = label
        self.cardinality: int = 0
        self.visited: bool = False
        self.moral_adjacency: list[Node] = []
        self.value: int = None


class Intervention:
    def __init__(self, label: str, value: int = None):
        self.label = label
        self.value = value

    def set_value(self, value: int):
        self.value = value


class Target:
    def __init__(self, label: str, value: int = None):
        self.label = label
        self.value = value

    def set_value(self, value: int):
        self.value = value
