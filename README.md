# MomPropEmbeddings

Python implementation of the moment propagation (MomProp) embeddings described in 
_Prediction of cancer driver genes through network-based moment propagation of mutation scores_
(Anja C. Gumpinger, Kasper Lage, Heiko Horn and Karsten Borgwardt), submitted to ISMB 2020.

## Requirements
To execute the python package, the following modules are required.
```
numpy
scipy
python-igraph
pandas
```

## Tests
Unittest can be found in `./tests`, and executed with `./tests/unittest/runAllTests.sh`.

## Moment Propagation Embeddings.
The generation of moment propagation embeddings is a four-step process:
1. k-hop path weights have to be computed (functionality in `path_weights.py`)
2. The data have to be represented in an igraph object (functionality in `basegraph.py`)
3. Neighborhood features are created using the igraph representation (functionality in `features.py`)
4. MomProp embeddings are generated (functionality in `data.py`)


## Example.
This is currently under construction. Examples will be added.
Examples of how to execute the above steps can be found in `./examples`
