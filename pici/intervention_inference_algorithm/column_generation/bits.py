from gurobipy import Var

from pici.graph.node import Node


class Bit:
    def __init__(self, gurobi_var: Var, sign: bool) -> None:
        self.gurobi_var = gurobi_var
        self.sign = sign

    def __eq__(self, other):
        if not isinstance(other, Node):
            raise TypeError(f"Cannot compare Bit with {type(other)}")
        return self.gurobi_var == other.gurobi_var

    def __hash__(self):
        return hash(self.gurobi_var)

    def __repr__(self):
        return f"Bit({self.gurobi_var.VarName!r})"


class BitProduct:
    def __init__(self) -> None:
        self.bit_list: list[Bit] = []
        self.coef: float = 0

    def add_bit(self, bit: Bit):
        self.bit_list.append(bit)

    def set_coef(self, coef: float):
        self.coef = coef
