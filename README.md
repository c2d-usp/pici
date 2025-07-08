Causal Reasoning (WIP)
=======================
## Table of Contents
1. [About](#about)
2. [Usage](#usage)
3. [How it works](#how-it-works)
4. [Developer Tools](#developer-tools)
   - [Install](#install)
   - [Virtual Environment](#virtual-environment)
   - [Running Examples](#running-examples)
   - [Linters](#linters)
     - [Black](#black)
   - [Unittest](#unittest)
5. [Acknowledgements](#acknowledgements)


## About

Causal Reasoning is a causal inference package that can handle Partially Identifiable Queries in Quasi-Markovian Structural Causal Models.

This project was based on the work of João Pedro Arroyo and João Gabriel on [GitHub](https://github.com/Causal-Inference-Group-C4AI/Linear-Programming-For-Interventional-Queries).


## Usage

- Install the package
```python
pip install causal_reasoning
```

- Import the package
```python
import causal_reasoning
```

- Create a causal model:
```python
df = pd.read_csv(model_csv_path)
edges = "Z -> X, X -> Y, U1 -> X, U1 -> Y, U2 -> Z"
custom_cardinalities = {"Z": 2, "X": 2, "Y": 2, "U1": 0, "U2": 0}
unobservable_variables = ["U1", "U2"]

model = causal_reasoning.causal_model.CausalModel(
  data=df,
  edges=edges,
  custom_cardinalities=custom_cardinalities,
  unobservables_labels=unobservable_variables,
)
```

- Set target and intervetions
```python
model.set_interventions([('X', 1)])
model.set_target(('Y', 1))
```

- Make the query
```python
lower_bound, upper_bound = model.inference_intervention_query()
```
Or you can pass the target and intervention as an argument: 

```python
lower_bound, upper_bound = model.inference_intervention_query([('X', 1)], ('Y', 1))
```

## How it works

TODO: EXPLAIN THE THEORY AND METHODS.

## Developer tools

All development tools are managed with [Poetry](https://python-poetry.org/docs/).  
To get started, install Poetry by following the [official instructions](https://python-poetry.org/docs/#installation), then activate the virtual environment.

### Install

- Install dependencies
```bash
poetry install
```


### Virtual Environment

Activate the virtual environment:

- Activate poetry virtual environment
```bash
eval $(poetry env activate)
```

- To exit the poetry virtual environment run:
```bash
deactivate
```

### Running Examples

Example:
```bash
python main.py
```

You also can run unit tests:
```bash
python tests/test_causal_model.py
```


### Linters

This project uses some linters to follow a code standardization that improves code consistency and cleanliness.

#### Black

This project uses **[Black](https://black.readthedocs.io/en/stable/)** for automatic Python code formatting.
Black is an code formatter that ensures consistency by enforcing a uniform style.

Usage example for a file:

```bash
black your_script.py
```

For all files in the current directory and sub-directories:

```bash
black .
```

Running this command will change automatically.



### Unittest

TODO: EXPLAIN SOME UNITTESTS


## Acknowledgements
We thank the ICTi, Instituto de Ciência e Tecnologia Itaú, for providing key funding
for this work through the C2D - Centro de Ciência de Dados at Universidade de São Paulo.

Any opinions, findings, conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of Itaú Unibanco and Instituto de Ciência e Tecnologia Itaú. All data used in this study comply with the Brazilian General Data Protection Law.
