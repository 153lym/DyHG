# DyHG

Fusing High-Order Topology Hypergraphs and Dynamic Semantics Network for Spatial Domain Identification

## Overview:

We introduce DyHG.


## Run environment:
 
DyHG is implemented in the pytorch framework. The detailed running environment can be found in file [DyHG.yaml](DyHG.yaml).
In the experiments, the GPU we used was NVIDIA RTX A6000.

## Run

### ST raw dataset download
The DLPFC dataset with 10x Visium platform is accessible within the spatialLIBD package (http://spatial.libd.org/spatialLIBD). 

The mouse & human liver (10X_MOUSE& 10X_HUMAN) dataset with 10x Visium platform is collected from https://www.livercellatlas.org/. 

The mouse medial prefrontal cortex (MMPC) dataset with STARmap platform is collected from http://clarityresourcecenter.org/. 

The MOSTA hypothalamic preoptic region with Stereo platform is collected from https://db.cngb.org/stomics/mosta/. 

### Run DyHG

We give example on the DLPFC datasets with 10x Visium platform. 
We provide the example code in [main_DLPFC.py](main_DLPFC.py).

`python main_DLPFC.py`

For different ST data, it is necessary to set the range of alpha and gamma parameters to find the optimal set of parameter values. 
During the experiment, we set the range of [0, 10] for both.

## Citation:
**This repository contains the source code for the paper:**

`Fusing High-Order Topology Hypergraphs and Dynamic Semantics Network for Spatial Domain Identification`
