[DyNAS-T Logo](docs/images/dynast_logo.png)

# DyNAS-T

DyNAS-T (**Dy**namic **N**eural **A**rchitecture **S**earch **T**oolkit) is SuperNet NAS
optimization package designed for finding the optimal Pareto front during neural architure
search while minimizing the number of evaulation measurements required. It supports
multi-objective and single-objective problems for a variety of domains supported by the
Intel AI Lab HANDI framework. Additionally, models can leverage the Intel OpenVINO package
to improve performance an Intel hardware. 

## Supported SuperNet Frameworks

DyNAS-T is intended to be used with existing standalone SuperNet frameworks suchs as Intel
HANDI, Intel BootstrapNAS, or external libraries such as Hanlab Once-for-All (OFA). 

* HANDI Image Classification (supported)
* HANDI Machine Translation (Q3'21)
* BootstarNAS (Q4'21)
* Hanlab OFA (Q4'21)

## Supported DNN Backends

DyNAS-T is intended to be used with existing standalone SuperNet frameworks suchs as Intel
HANDI, Intel BootstrapNAS, or external libraries such as Hanlab Once-for-All (OFA). 

* HANDI Image Classification (supported)
* HANDI Machine Translation (Q3'21)
* BootstarNAS (Q4'21)
* Hanlab OFA (Q4'21)

## Design Overview

DyNAS-T supplements existing SuperNet Training frameworks in the following ways. 

[DyNAS-T Design Flow](docs/images/dynast_design.png)

## Requirements

Requirements for using DyNAS-T are given in `requirements.txt`. Key requirements to
be aware of are:  
* pymoo == 0.4.2   
* optuna == 2.4.0   
* scikit-learn == 0.23.2  
* pandas == 1.1.5  

