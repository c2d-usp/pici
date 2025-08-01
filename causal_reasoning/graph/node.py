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
        self.target_value: int = None
        self.intervention_value: int = None
