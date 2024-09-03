# Enhanced Multi-Objective, Performance-Driven De novo Design of Polyimides Leveraging Deep Reinforcement Learning
***


PI_RL: Enhanced Multi-Objective, Performance-Driven De novo Design of Polyimides Leveraging Deep Reinforcement Learning
- https://xxxxxx.xxx

## Background
***
This code is the basis of our work submitted to *Journal of Chemical Information and Modeling*, aiming to 
integrate *a-priori* knowledge, i.e. group contribution theory, with graph neural networks and attention mechanism, and bring more insights in the prediction of thermodynamic properties. This is an **alpha version** of the models used to generate the resuls published in:

[Enhanced Multi-Objective, Performance-Driven De novo Design of Polyimides Leveraging Deep Reinforcement Learning](https://xxxxxx.xxx)

## Prerequisites and dependencies
```
$ env.yml
```
## Usage
***
### Raw data of PI structure and properties
The relevant files are kept in './raw_data'

### Descriptor model
The relevant files and code for the descriptor model are in '.QSPR/Descriptors'

A command line for preprocessing descriptor conversion and dimensionality reduction:
```commandline
$ python data_process.py 
```
The result files after propocessing are in '.QSPR/Descriptors/datasets/'.

To carry out a single-objective bayesian optimization on a descriptor model, do:
```commandline
$ python Optimization.py
```
To generate ensemble models with random initializations on descriptor models, do:
```commandline
$ python Ensembling.py
```

### polyBERT model
The relevant files and code for the descriptor model are in '.QSPR/polyBERT'

To carry out a single-objective bayesian optimization on a descriptor model, do:
```commandline
$ python Optimization.py
```
To generate ensemble models with random initializations on descriptor models, do:
```commandline
$ python Ensembling.py
```
A command line for predicting each PI in datasets:
```commandline
$ python Ensemble_PI.py
```


## Usage
Set python path: `export PYTHONPATH=.`.

### Decomposition
In decomposition step, molecules are decomposed into subgraphs by gSpan. You can select raw or junction tree data.

```python
from pathlib import Path
from mi_collections.moldr.decompose import MolsMining, DefaultConfig
from mi_collections.chemutils import get_mol

test_smiles = [
    "CC1CCC2=CC=CC=C2O1",
    "CC",
    "COC",
    "c1ccccc1",
    "CC1C(=O)NC(=O)S1",
    "CO",
]
mols = [get_mol(s) for s in test_smiles]
minsup = int(len(mols) * 0.1)
config = DefaultConfig(
    data_path=Path("zinc_jt.data"), support=minsup, lower=2, upper=7, method="jt"
)
runner = MolsMining(config)
gspan = runner.decompose(mols)
```

If you want to see the subgraphs in detail, see `examples/decomponsition.ipynb`.

### Reassembing
- Training 
The agent takes an action from building blocks obtained from a decomposition step. The agent is trained by PPO with RLlib.

```shell
python train.py --epochs 100 --num_workers 128 --num_gpus 1
```

Generated molecules are sampled through the trained agent. Select and rewrite the model path that you want to use.

```shell
python run_moldr.py 
```
