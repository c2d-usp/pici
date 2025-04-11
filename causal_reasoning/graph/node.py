from typing import Generic, TypeVar


T = TypeVar("T")


class Node(Generic[T]):
    def __init__(
        self,
        children: list["Node"],
        parents: list["Node"],
        latentParent: "Node",
        isLatent: bool,
        label: T,
        cardinality: int,
    ):
        self.children: list["Node"] = children
        self.parents: list["Node"] = parents
        self.latentParent: "Node" = latentParent
        self.isLatent: bool = isLatent
        self.label: T = label
        self.cardinality: int = cardinality
        self.visited: bool = False
        self.moral_adjacency: list[Node] = []
        self.value: int = None
