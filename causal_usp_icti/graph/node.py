class Node:
    children: list[int]
    parents: list[int]
    latentParent: int
    isLatent: bool

    def __init__(self, children, parents, latentParent, isLatent):
        self.children = children
        self.parents = parents
        self.latentParent = latentParent
        self.isLatent = isLatent
