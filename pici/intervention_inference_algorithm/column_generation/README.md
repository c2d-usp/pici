COLUMN GENERATION ALGORITHM -- Work In Progress
 
## Theory

## Algorithm

For an intervention query in a causal model, we consider a subset of nodes of the causal model graph called W.
It contains the nodes needed to make the intervention query.

To create the optimization problem we create variables for each node in W given its parents realization.

Let's consider the graph: "Z -> X, X -> Y, U -> X, U-> Y", where the cardinalities of the variables are Z=4, X=3, and Y=2.
W = "X", "Y"

For node X, we consider its endogenous parents: "Z"

For each realization of its parents we have a set of variables of X. The set of variables represents bits that configure the cardinality of X. So the number of bits are k = ceil(log2(X cardinality))
"Z" |  Variables
-------------
0   |  V0x_0, V0x_1, ... , V0x_{k-1}
1   |  V1x_0, V1x_1, ... , V1x_{k-1}
2   |  V2x_0, V2x_1, ... , V2x_{k-1}
3   |  V3x_0, V3x_1, ... , V3x_{k-1}
