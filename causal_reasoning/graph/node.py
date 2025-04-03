from typing import Any
from typing import TypeVar, Generic

T = TypeVar("T")

class Node(Generic[T]):
    children: list[T]
    parents: list[T]
    latentParent: T
    isLatent: bool
    value: T
    cardinality: int
    visited: bool

    def __init__(self, children, parents, latentParent, isLatent, value: T):
        self.children = children
        self.parents = parents
        self.latentParent = latentParent
        self.isLatent = isLatent
        self.value = value
        self.cardinality = 0
        self.visited: bool = False


class Intervention:
    label: str
    value: int

    def __init__(self, label: str, value: int=None):
        self.label = label
        self.value = value

    def set_value(self, value: int):
        self.value = value


class Target:
    label: str
    value: int

    def __init__(self, label: str, value: int=None):
        self.label = label
        self.value = value

    def set_value(self, value: int):
        self.value = value