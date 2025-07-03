Causal Reasoning (WIP)
=======================
## Table of Contents
1. [About](#about)  
2. [Usage](#usage)  
3. [How it works](#how-it-works)  
4. [Developer Tools](#developer-tools)  
   - [Virtual Environment](#virtual-environment)  
   - [Running Examples](#running-examples)  
   - [Linters](#linters)  
     - [Black](#black)  
     - [Isort](#isort)  
   - [Unittest](#unittest)  
5. [Acknowledgements](#acknowledgements)


## About

Causal Reasoning is a causal inference package that can handle Partially Identifiable Queries in Quasi-Markovian Structural Causal Models.

This project was based on the work of João Pedro Arroyo and João Gabriel on [GitHub](https://github.com/Causal-Inference-Group-C4AI/Linear-Programming-For-Interventional-Queries).


## Usage

TODO: SHOW HOW TO USE THE PACKAGE WITH PYPI.

## How it works

TODO: EXPLAIN THE THEORY AND METHODS.

## Developer tools

All development tools are managed with [Poetry](https://python-poetry.org/docs/).  
To get started, install Poetry by following the [official instructions](https://python-poetry.org/docs/#installation), then activate the virtual environment.

### Virtual Environment

Activate the virtual environment:

- (On Linux) Activate poetry virtual environment
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

Usage Example:

```bash
black your_script.py
```

Running this command will change automatically.


#### Isort

**isort** focuses specifically on the organization of import statements.
It automatically sorts imports alphabetically and separates them into sections (standard library, third-party, and local imports).

Usage Example:

```bash
isort your_script.py
```

After running the command, save the file to apply the sorted imports.


### Unittest

TODO


## Acknowledgements
We thank the ICTi, Instituto de Ciência e Tecnologia Itaú, for providing key funding
for this work through the C2D - Centro de Ciência de Dados at Universidade de São Paulo.

Any opinions, findings, conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of Itaú Unibanco and Instituto de Ciência e Tecnologia Itaú. All data used in this study comply with the Brazilian General Data Protection Law.
