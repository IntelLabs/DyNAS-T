![DyNAS-T Logo](docs/images/dynast_logo.png)

# DyNAS-T

DyNAS-T (**Dy**namic **N**eural **A**rchitecture **S**earch **T**oolkit) is a SuperNet NAS
optimization package designed for finding the optimal Pareto front during neural architure
search while minimizing the number of search validation measurements. It supports
single-/multi-/many-objective problems for a variety of domains supported by the
Intel AI Lab [HANDI framework](https://github.com/intel-innersource?q=handi&type=all&language=&sort=). The system currently heavily utilizes the [pymoo](https://pymoo.org/)
optimization library. Some of the key DyNAS-T features are:
* Automatic handling of supernetwork parameters for search and predictor training
* Genetic Algorithm (e.g., NSGA-II) multi-objective subnetworks
* ConcurrentNAS accelerated search using approximate predictors
* Warm-start (transfer) search
* Search population statistical analysis

## Supported SuperNet Frameworks

DyNAS-T is intended to be used with existing standalone SuperNet frameworks suchs as Intel
HANDI, [Intel BootstrapNAS](https://gitlab.devtools.intel.com/jpmunoz/bootstrapnas_poc_subnet_extraction), or external libraries such as [Once-for-All (OFA)](https://github.com/mit-han-lab/once-for-all).

* HANDI MobileNetV3 (supported)
* HANDI ResNet50 (supported)
* HANDI Transformer (supported)
* HANDI Recommender (Q1'22)
* BootstrapNAS ResNet50 torchvision (supported)

## Getting Started

To setup DyNAS-T run `pip install -e .` or make a local copy of the `dynast` subfolder in your
local subnetwork repository with the `requirements.txt` dependencies installed.

Examples of setting up DyNAS-T with various SuperNet frameworks are given in the
./examples directory. We suggested using `dynast_mbnv3_full.py` as a starting point
using the HANDI MobileNetV3 supernetwork.

## Design Overview

DyNAS-T supplements existing SuperNet Training frameworks in the following ways.

![DyNAS-T Design Flow](docs/images/dynast_design.png)

## Release Notes

0.1.0 - 0.3.0:
* Updated to pymoo version 0.5.0
* Added example templates for HANDI MobileNetV3, Transformer, ResNet50
* Added ConcurrentNAS examples for HANDI MobileNetV3, BootstrapNAS
* Added OpenVINO INT8 search example for HANDI MobileNetV3
* Updated search results to be managed by pymoo results object.
* New `ParameterManager` handles tranlation between dictionary, pymoo, and one-hot vector formats

