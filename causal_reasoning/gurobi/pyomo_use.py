from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    ConstraintList,
    SolverFactory,
    value,
    NonNegativeReals,
)


def solve_bounds_with_pyomo(
    objFunctionCoefficients: list[float],
    decisionMatrix: list[list[int]],
    probs: list[float],
    intervals: list[tuple[int, int]],
):
    num_vars = len(objFunctionCoefficients)
    num_constraints = len(decisionMatrix)

    def build_model(obj_coeffs):
        model = ConcreteModel()

        # Variables with bounds from intervals
        model.x = Var(range(num_vars), domain=NonNegativeReals)
        for i in range(num_vars):
            lb, ub = intervals[i]
            model.x[i].setlb(lb)
            model.x[i].setub(ub)

        # Objective
        model.obj = Objective(
            expr=sum(obj_coeffs[i] * model.x[i] for i in range(num_vars)), sense=1
        )

        # Equality constraints
        model.constraints = ConstraintList()
        for j in range(num_constraints):
            expr = sum(decisionMatrix[j][i] * model.x[i] for i in range(num_vars))
            model.constraints.add(expr == probs[j])

        return model

    # Lower bound (minimize)
    model_min = build_model(objFunctionCoefficients)
    solver = SolverFactory("gurobi")
    result_min = solver.solve(model_min, tee=False)
    lowerBound = value(model_min.obj)

    # Upper bound (maximize)
    model_max = build_model([-c for c in objFunctionCoefficients])
    result_max = solver.solve(model_max, tee=False)
    upperBound = -value(model_max.obj)

    return lowerBound, upperBound
