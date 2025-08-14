PICI -- Partially Identifiable Causal Inference
=======================
## Table of Contents
1. [About](#about)
2. [Usage](#usage)
   - [Install and Import](#install-and-import)
   - [Causal Model Creation](#causal-model-creation)
   - [Interventional Query](#interventional-query)
   - [Identifiability Checker](#identifiability-checker)
   - [PN & PS Approximations](#pn-and-ps-approximations)
3. [Theory](#theory-behind)
4. [Developer Tools](#developer-tools)
   - [Install](#install)
   - [Virtual Environment](#virtual-environment)
   - [Running Examples](#running-examples)
   - [Linters](#linters)
     - [Black](#black)
     - [Ruff](#ruff)
5. [Acknowledgements](#acknowledgements)


## About

PICI stands for Partially Identifiable Causal Inference.
It is a causal inference package that can handle discrete Partially Identifiable Queries in Quasi-Markovian Structural Causal Models.

This project was based on the work of João Pedro Arroyo and João Gabriel Rodrigues on [GitHub](https://github.com/Causal-Inference-Group-C4AI/Linear-Programming-For-Interventional-Queries).


## Usage

### Install and import
- Install the package.
```bash
pip install pici
```

- Import the package.
```python
import pici
```

### Causal model creation

- Create a causal model:
```python
df = pd.read_csv(model_csv_path)
edges = "Z -> X, X -> Y, U1 -> X, U1 -> Y, U2 -> Z"
unobservable_variables = ["U1", "U2"]
custom_cardinalities = {"Z": 5, "X": 2, "Y": 16, "U1": 0, "U2": 0}
interventions=[('X',0)]
target=('Y',0)

model = pici.CausalModel(
  data=df,
  edges=edges,
  custom_cardinalities=custom_cardinalities,
  unobservables_labels=unobservable_variables,
  interventions=interventions,    # Optional in the model creation
  target=target,                  # Optional in the model creation
)
```

- You can set the target and interventions:
```python
model.set_interventions([('X', 1)])
model.set_target(('Y', 1))
```

### Interventional query

- You can make the query:
```python
model.intervention_query()
```

Or you can pass the target and interventions as arguments:

```python
model.intervention_query([('X', 1)], ('Y', 1))
```
- When `intervention_query` is called, it automatically checks if your query is identifiable or partially identifiable.
- If your query is **identifiable**, the return value is a `string`.

```python
exact_value = model.intervention_query()
```

- If your query is **partially identifiable**, the return value is a `tuple` of `strings`.
```python
lower_bound, upper_bound = model.intervention_query()
```

### Identifiability checker

- You can check if your intervention is identifiable by Backdoor criterion, Frontdoor criterion, or [ID algorithm](https://cdn.aaai.org/AAAI/2006/AAAI06-191.pdf):

```python
edges = "U1 -> X, U1 -> Y, X -> W, W -> Y, U2 -> W"
unobservables_labels=["U1", "U2"]

model = CausalModel(
    data=df,
    edges=edges,
    unobservables_labels=unobservables_labels
)

is_identifiable, identifiable_method, additional_detail = model.is_identifiable_intervention(
    interventions=[("X", 0)], target=("Y", 1)
)
```

### PN and PS approximations

- You can make PN and PS approximations, which we call *weak-pn* and *weak-ps*. These approximations are based on the paper **Probabilities of Causation and Root Cause Analysis with Quasi-Markovian Models** by Laurentino et al., 2025 (public link soon).

The same way as the `intervention_query`, if it's identifiable it will return a `string` otherwise a `tuple` of `strings`.

```python
pn_value = model.weak_pn_inference(intervention_label='X', target_label='Y')
ps_value = model.weak_ps_inference(intervention_label='X', target_label='Y')
```

## Theory behind

If you want to understand the theory behind our approach, you can read the paper [Arroyo et al., 2025](https://openreview.net/forum?id=aUPT1kEiwP).


## Developer tools

All development tools are managed with [Poetry](https://python-poetry.org/docs/).  
To get started, install Poetry by following the [official instructions](https://python-poetry.org/docs/#installation), then activate the virtual environment.

### Install

- Install dependencies. Once you have installed Poetry, you can install the required packages:
```bash
poetry install
```


### Virtual Environment

Activate the virtual environment:

- Activate the Poetry virtual environment
```bash
eval $(poetry env activate)
```

- To exit the Poetry virtual environment, run:
```bash
deactivate
```

### Running Examples

Example:
```bash
python3 examples/example.py
```

- You also can run unit tests:
```bash
python3 tests/run_all_tests.py
```


### Linters

This project uses some linters to follow a code standardization that improves code consistency and cleanliness.

#### Ruff

This project uses **[Ruff](https://github.com/astral-sh/ruff)** to remove unused imports and sort them.

Usage example for a file:

```bash
ruff check your_file.py --fix -s
```

For all files in the current directory and sub-directories:

```bash
ruff check . --fix -s
```

Running this command will make changes automatically.
We suggest using the flag `-s` to silence the numerous print logs.


#### Black

This project uses **[Black](https://black.readthedocs.io/en/stable/)** for automatic Python code formatting.
Black is a code formatter that ensures consistency by enforcing a uniform style.

Usage example for a file:

```bash
black your_file.py
```

For all files in the current directory and sub-directories:

```bash
black .
```

Running this command will change automatically.


## Acknowledgements
We thank the ICTi, Instituto de Ciência e Tecnologia Itaú, for providing key funding
for this work through C2D - Centro de Ciência de Dados at Universidade de São Paulo.

Any opinions, findings, conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of Itaú Unibanco and Instituto de Ciência e Tecnologia Itaú. All data used in this study comply with the Brazilian General Data Protection Law.
