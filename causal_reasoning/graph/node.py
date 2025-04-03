from typing import Any
from typing import TypeVar, Generic

T = TypeVar("T")

class Node(Generic[T]):
    children: list[str]
    parents: list[str]
    latentParent: str
    isLatent: bool
    value: T

    def __init__(self, children, parents, latentParent, isLatent, value: T):
        self.children = children
        self.parents = parents
        self.latentParent = latentParent
        self.isLatent = isLatent
        self.value = value


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