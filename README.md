# MoProEmbeddings

Python implementation of the moment propagation (MoPro) embeddings described in 
_Prediction of cancer driver genes through network-based moment propagation of mutation scores_
(Anja C. Gumpinger, Kasper Lage, Heiko Horn and Karsten Borgwardt). 
See https://academic.oup.com/bioinformatics/article/doi/10.1093/bioinformatics/btaa581/5861532 
for the publication.

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
IPython notebooks that show how to execute the above steps can be found in `./examples`. First, the data has to be processed as shown in `./examples/preprocessing.ipynb`. Next, the MomProp embeddings can be computed, as explained in `mom_prop_embeddings.ipynb`.
