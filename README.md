# Enhanced Multi-Objective, Performance-Driven De novo Design of Polyimides Leveraging Deep Reinforcement Learning
***


PI_RL: Enhanced Multi-Objective, Performance-Driven De novo Design of Polyimides Leveraging Deep Reinforcement Learning
- https://xxxxxx.xxx

## Background
***
This code is the basis of our work submitted to *XXXXXXX*, which aims to *de novo* design of polyimides using PPO algorithm of reinforcement learning based on fragment generation to bring more insights into polymer design. 

## Prerequisites and dependencies
```
$ env.yml
```
## Usage
***
### Raw data of PI structure and properties
The relevant files are kept in './raw_data'

### GNN model
The relevant files and code for the descriptor model are in './QSPR/GNN/'

A command line for training a AFP model and tuning the splitting seed:
```commandline
$ python Seed_Tuning.py 
```
A command line for training a FraGAT model and tuning the splitting seed:
```commandline
$ python Seed_Tuning_Frag.py 
```
The results of every attempt containing metrics on three folds are automatically printed in './QSPR/GNN/output/'.

To carry out a single-objective bayesian optimization on a AFP model, do:
```commandline
$ python Optimization.py
```
To carry out a single-objective bayesian optimization on a FraGAT model, do:
```commandline
$ python Optimization_frag.py
```
To generate ensemble models with random initializations on AFP models, do:
```commandline
$ python Ensembling.py
```
To generate ensemble models with random initializations on FraGAT models, do:
```commandline
$ python Ensembling_frag.py
```
A command line for predicting each PI in datasets:
```commandline
$ python Ensemble_PI.py
$ python Ensemble_frag_PI.py
```

### Descriptor model
The relevant files and code for the descriptor model are in './QSPR/Descriptors'

A command line for preprocessing descriptor conversion and dimensionality reduction:
```commandline
$ python data_process.py 
```
The result files after propocessing are in './QSPR/Descriptors/datasets/'.

To carry out a single-objective bayesian optimization on a descriptor model, do:
```commandline
$ python Optimization.py
```
To generate ensemble models with random initializations on descriptor models, do:
```commandline
$ python Ensembling.py
```

### polyBERT model
The relevant files and code for the descriptor model are in './QSPR/polyBERT'
#### Before running, download polyBERT model on https://huggingface.co/kuelumbus/polyBERT
To carry out a single-objective bayesian optimization on a polyBERT model, do:
```commandline
$ python Optimization.py
```
To generate ensemble models with random initializations on polyBERT models, do:
```commandline
$ python Ensembling.py
```
A command line for predicting each PI in datasets:
```commandline
$ python Ensemble_PI.py
```

### RL_PPO
To decompose dianhydride and diamine, do:
```commandline
$ python Decompose.py
```
The building blocks of dianhydride and diamine are in './RL_PPO/outpus/building_blocks/'

The agent takes an action from building blocks obtained from a decomposition step. The agent is trained by PPO with RLlib.
```commandline
$ python train.py
```
Generated polyimides are sampled through the trained agent. Select and rewrite the model path that you want to use.
To generate polyimides randomly, do:
```commandline
$ python env_test.py
```
To the cluster analysis of the generate polyimides, run './RL_PPO/outpus/postprocess.py'
