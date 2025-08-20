from __future__ import annotations


class Node:
    def __init__(
        self,
        children: list["Node"],
        parents: list["Node"],
        latentParent: "Node",
        isLatent: bool,
        label: str,
        cardinality: int,
    ):
        self.children: list["Node"] = children
        self.parents: list["Node"] = parents
        self.latentParent: "Node" = latentParent
        self.isLatent: bool = isLatent
        self.label: str = label
        self.cardinality: int = cardinality
        self.visited: bool = False
        self.value: int = None
        self.intervened_value: int = None

    def __eq__(self, other):
        if not isinstance(other, Node):
            raise TypeError(f"Cannot compare Node with {type(other)}")
        return self.label == other.label

    def __hash__(self):
        return hash(self.label)

    def __repr__(self):
        return f"Node({self.label!r})"
