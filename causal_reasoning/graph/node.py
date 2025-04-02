from typing import Any

class Node:
    children: list[str]
    parents: list[str]
    latentParent: str
    isLatent: bool
    value: Any

    def __init__(self, children, parents, latentParent, isLatent):
        self.children = children
        self.parents = parents
        self.latentParent = latentParent
        self.isLatent = isLatent

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