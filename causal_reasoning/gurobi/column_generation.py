import numpy as np
import gurobipy as gp
from gurobipy import GRB


# https://colab.research.google.com/github/Gurobi/modeling-examples/blob/master/colgen-cutting_stock/colgen-cutting_stock.ipynb#scrollTo=HXWEmpkxUIR_


class MasterProblem:
    def __init__(self):
        self.model = gp.Model("master")
        self.vars = None
        self.constrs = None

    def setup(self, patterns, demand):
        num_patterns = len(patterns)
        # Variáveis Mecanismos
        # self.vars = self.model.addVars(mechanisms, ...)
        self.vars = self.model.addVars(num_patterns, obj=1, name="Pattern")

        # Constraints
        # self.constrs = self.model.addConstrs(decisionMatrix == probs)
        self.constrs = self.model.addConstrs(
            (
                gp.quicksum(
                    patterns[pattern][piece] * self.vars[pattern]
                    for pattern in range(num_patterns)
                )
                >= demand[piece]
                for piece in demand.keys()
            ),
            name="Demand",
        )
        self.model.modelSense = GRB.MINIMIZE
        # Turning off output because of the iterative procedure
        self.model.params.outputFlag = 0
        self.model.update()

    def update(self, pattern, index):
        new_col = gp.Column(coeffs=pattern, constrs=self.constrs.values())
        self.vars[index] = self.model.addVar(
            obj=1, column=new_col, name=f"Pattern[{index}]"
        )
        self.model.update()


class SubProblem:
    def __init__(self):
        self.model = gp.Model("subproblem")
        self.vars = {}
        self.constr = None

    def setup(self, stock_length, lengths, duals):
        self.vars = self.model.addVars(
            len(lengths), obj=duals, vtype=GRB.INTEGER, name="Frequency"
        )
        self.constr = self.model.addConstr(
            self.vars.prod(lengths) <= stock_length, name="Knapsack"
        )
        self.model.modelSense = GRB.MAXIMIZE
        # Turning off output because of the iterative procedure
        self.model.params.outputFlag = 0
        # Stop the subproblem routine as soon as the objective's best bound becomes
        # less than or equal to one, as this implies a non-negative reduced cost for
        # the entering column.
        self.model.params.bestBdStop = 1
        self.model.update()

    def update(self, duals):
        self.model.setAttr("obj", self.vars, duals)
        self.model.update()


class CuttingStock:
    def __init__(self, stock_length, pieces):
        self.stock_length = stock_length
        self.pieces, self.lengths, self.demand = gp.multidict(pieces)
        self.patterns = None
        self.duals = [0] * len(self.pieces)
        piece_reqs = [length * req for length, req in pieces.values()]
        self.min_rolls = np.ceil(np.sum(piece_reqs) / stock_length)
        self.solution = {}
        self.master = MasterProblem()
        self.subproblem = SubProblem()

    def _initialize_patterns(self):
        # Find trivial patterns that consider one final piece at a time,
        # fitting as many pieces as possible into the stock material unit
        patterns = []
        for idx, length in self.lengths.items():
            pattern = [0] * len(self.pieces)
            pattern[idx] = self.stock_length // length
            patterns.append(pattern)
        self.patterns = patterns

    def _generate_patterns(self):
        self._initialize_patterns()
        self.master.setup(self.patterns, self.demand)
        self.subproblem.setup(self.stock_length, self.lengths, self.duals)
        while True:
            self.master.model.optimize()
            self.duals = self.master.model.getAttr("pi", self.master.constrs)
            self.subproblem.update(self.duals)
            self.subproblem.model.optimize()
            reduced_cost = 1 - self.subproblem.model.objVal
            if reduced_cost >= 0:
                break

            pattern = [0] * len(self.pieces)
            for piece, var in self.subproblem.vars.items():
                if var.x > 0.5:
                    pattern[piece] = round(var.x)
            self.master.update(pattern, len(self.patterns))
            self.patterns.append(pattern)

    def solve(self):
        """
        Gurobi does not support branch-and-price, as this requires to add columns
        at local nodes of the search tree. A heuristic is used instead, where the
        integrality constraints for the variables of the final root LP relaxation
        are installed and the resulting MIP is solved. Note that the optimal
        solution could be overlooked, as additional columns are not generated at
        the local nodes of the search tree.
        """
        self._generate_patterns()
        self.master.model.setAttr("vType", self.master.vars, GRB.INTEGER)
        self.master.model.optimize()
        for pattern, var in self.master.vars.items():
            if var.x > 0.5:
                self.solution[pattern] = round(var.x)
