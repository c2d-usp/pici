import os
import time
from statistics import mean

import networkx as nx
import pandas as pd

from pici.causal_model import CausalModel
from experiments.utils.scalable_graphs_helper import generate_two_latents_scalable_string_edges, generate_two_latents_binary_scalable_cardinalities


THIS_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
import sys
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Tamanho do Problema
m = 1
n = 3

print("----------------------------------------------------------------------------------------------")
print(f"N = {n} and M = {m}")
edges = generate_two_latents_scalable_string_edges(N=n, M=m)
cardinalities = generate_two_latents_binary_scalable_cardinalities(N=n, M=m)
unobs_vars = ["U1", "U2"]
target = "Y"
target_value = 1
intervention = "X"
intervention_value = 1
# Load Data
csv_path = f"data/csv/n{n}_m{m}_scaling_case.csv"
df = pd.read_csv(csv_path)

# Create Causal Model
model = CausalModel(
    data=df,
    edges=edges,
    custom_cardinalities=cardinalities,
    unobservables_labels=unobs_vars,
    interventions=(intervention, intervention_value),
    target=(target, target_value),
    #optimization_algorithm="column_gen" # Caso precise testar a solução II
    optimization_algorithm="bit_solution", # "bit_solution" é solução III
    #max_time=2000
)

# Calculating the interventions
start_time = time.perf_counter()
lower, upper = model.intervention_query()
end_time = time.perf_counter()
print(
    f"{lower} <= P({target}={target_value}|do({intervention}={intervention_value})) <= {upper}"
)
