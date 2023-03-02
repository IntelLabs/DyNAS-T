![DyNAS-T Logo](https://github.com/IntelLabs/DyNAS-T/blob/main/docs/images/dynast_logo.png?raw=true)

# DyNAS-T

DyNAS-T (**Dy**namic **N**eural **A**rchitecture **S**earch **T**oolkit) is a super-network neural architecture
search NAS optimization package designed for efficiently discovering optimal deep neural network (DNN)
architectures for a variety of performance objectives such as accuracy, latency, multiply-and-accumulates,
and model size.

## Background

Neural architecture search, the study of automating the discovery of optimal deep neural network architectures for tasks in domains such as computer vision and natural language processing, has seen rapid growth in the machine learning research community. The computational overhead of evaluating DNN architectures during the neural architecture search process can be very costly due to the training and validation cycles. To address the training overhead, novel weight-sharing approaches known as one-shot or super-networks [1] have offered a way to mitigate the training overhead by reducing training times from thousands to a few GPU days. These approaches train a task-specific super-network architecture with a weight-sharing mechanism that allows the sub-networks to be treated as unique individual architectures. This enables sub-network model extraction and validation without a separate training cycle.

To learn more about super-networks and how to define/train them, please see our [super-network tutorial](notebooks/Supernet_Tutorial.ipynb).

## Algorithms

Evolutionary algorithms, specifically genetic algorithms, have a history of usage in NAS and continue to gain popularity as a highly efficient way to explore the architecture objective space. DyNAS-T supports a wide range of evolutionary algorithms (EAs) such as NSGA-II [2] by leveraging the [pymoo](https://pymoo.org/) library.

A unique capability of DyNAS-T is the Lightweight Iterative NAS (LINAS) that pairs evolutionary algorithms with lightly trained objective predictors in an iterative cycle to accelerate architectural exploration [3]. This technique is ~4x more sample efficient than typical one-shot predictor-based NAS approaches.

![DyNAS-T Design Flow](https://github.com/IntelLabs/DyNAS-T/blob/main/docs/images/dynast_flow.png?raw=true)

The following number of optimization algorithms are supported by DyNAS-T in both standard and LINAS formats.

| 1 Objective<br>(Single-Objective) | 2 Objectives<br>(Multi-Objective) | 3 Objectives<br>(Many-Objective) |
|------------------|-----------------|----------------|
| GA* `'ga'`   | NSGA-II* `'nsga2'` | UNSGA-II* `'unsga3'`     |
| CMA-ES `'cmaes'` | AGE-MOEA `'age'` | CTAEA `'ctaea'`         |
|        |          | MOEAD `'moead'`          |
*Recommended for stability of search results

## Super-networks
DyNAS-T included support for the following super-network frameworks suchs as [Once-for-All (OFA)](https://github.com/mit-han-lab/once-for-all).

| Super-Network | Model Name | Dataset | Objectives/Measurements Supported |
|------------------|-----------------|-----------------|-----------------|
|OFA MobileNetV3-w1.0 | ofa_mbv3_d234_e346_k357_w1.0 | [ImageNet 1K](https://huggingface.co/datasets/imagenet-1k) | `accuracy_top1`, `macs`, `params`, `latency` |
|OFA MobileNetV3-w1.2 | ofa_mbv3_d234_e346_k357_w1.2 | [ImageNet 1K](https://huggingface.co/datasets/imagenet-1k) | `accuracy_top1`, `macs`, `params`, `latency` |
|OFA ResNet50 | ofa_resnet50 | [ImageNet 1K](https://huggingface.co/datasets/imagenet-1k) | `accuracy_top1`, `macs`, `params`, `latency` |
|OFA ProxylessNAS | ofa_proxyless_d234_e346_k357_w1.3 | [ImageNet 1K](https://huggingface.co/datasets/imagenet-1k) | `accuracy_top1`, `macs`, `params`, `latency` |
|TransformerLT | transformer_lt_wmt_en_de | WMT En-De | `bleu` (BLEU Score), `macs`, `params`, `latency` |
|BERT-SST2 | bert_base_sst2 | [SST2](https://huggingface.co/datasets/sst2) | `latency`, `macs`, `params`, `accuracy_sst2` |


> **_ImageNet:_**  When using any of the OFA super-networks, the ImageNet directory tree should have a separate directory for each of the classes in both `train` and `val` sets. To prepare your ImageNet dataset for use with OFA you could follow instructions available [here](https://jkjung-avt.github.io/ilsvrc2012-in-digits/).

> **_WMT En-De:_** To obtain and prepare dataset please follow instructions available [here](https://github.com/mit-han-lab/hardware-aware-transformers).

## Intel Library Support
The following software libraries are compatible with DyNAS-T:
* [Intel Neural Compressor (INC)](https://github.com/intel/neural-compressor/blob/master/examples/notebook/dynas/MobileNetV3_Supernet_NAS.ipynb)
* [Intel OpenVINO NNCF BootstrapNAS](https://github.com/openvinotoolkit/nncf/blob/develop/nncf/experimental/torch/nas/bootstrapNAS/BootstrapNAS.md) (Work-in-progress)

# Getting Started

To setup DyNAS-T from source code run `pip install -e .` or make a local copy of the `dynast` subfolder in your
local subnetwork repository with the `requirements.txt` dependencies installed.

You can also install DyNAS-T from PyPI:

```bash
pip install dynast
```


## Running DyNAS-T
The `run_search.py` template provide a starting point for running the NAS process. An evaluation is the process of determining the fitness of an architectural candidate. A *validation* evaluation is the costly process of running the full validation set. A *predictor* evaluation uses a pre-trained performance predictor.

* `supernet` - Name of the pre-trained super-network. See list of supported super-networks. For a custom super-network, you will have to modify the code including the `dynast_manager.py` and `supernetwork_registry.py` files.
* `optimization_metrics` - These are the metrics that the NAS process optimizes for. Note that the number of objectives you specify must be compatible with the supporting algorithm.
* `measurements` - In addition to the optimization metrics, you can specify which measurements you would like to take during an full evaluation.
* `search_tactic` - `linas` Lightweight iterative NAS (recommended) or `evolutionary` (good for benchmarking and testing new super-networks).
* `search_algo` - Determines which evolutionary algorithm to run for the `linas` low-fidelity inner loop or the `evolutionary` search tactic.
* `num_evals` - Number of evaluations (full validation measurements) to take. For example, if 1 validation measurement takes 5 minutes, 120 evaluations would take 10 hours.
* `seed` - Random seed.
* `population` - The size of the pool of candidates for each evolutionary generation. *50* is recommended for most cases, though this can be treated as a tunable hyperparameter.
* `results_path` - The location of the csv file that store information of the DNN candidates during the search process. The csv file is used for plotting NAS results.
* `dataset_path` - Location of the dataset used for training the super-network of interest.

### Single-Objective

*Example 1a.* NAS process for the OFA MobileNetV3-w1.0 super-network that optimizes for ImageNet Top-1 accuracy using a simple evolutionary genetic algorithm (GA) approach.

```bash
python run_search.py \
    --supernet ofa_mbv3_d234_e346_k357_w1.0 \
    --optimization_metrics accuracy_top1 \
    --measurements accuracy_top1 macs params \
    --results_path mbnv3w10_ga_acc.csv \
    --search_tactic evolutionary \
    --num_evals 250 \
    --search_algo ga
```

*Example 1b.* NAS process for the OFA MobileNetV3-w1.2 super-network that optimizes for ImageNet Top-1 accuracy using a LINAS + GA approach.

```bash
python run_search.py \
    --supernet ofa_mbv3_d234_e346_k357_w1.2 \
    --optimization_metrics accuracy_top1 \
    --measurements accuracy_top1 macs params \
    --results_path mbnv3w12_linasga_acc.csv \
    --search_tactic linas \
    --num_evals 250 \
    --search_algo ga
```

### Multi-Objective

*Example 2a.* NAS process for the OFA MobileNetV3-w1.0 super-network that optimizes for ImageNet Top-1 accuracy *and* multiply-and-accumulates (MACs) using a LINAS+NSGA-II approach.

```bash
python run_search.py \
    --supernet ofa_mbv3_d234_e346_k357_w1.0 \
    --optimization_metrics accuracy_top1 macs \
    --measurements accuracy_top1 macs params \
    --results_path mbnv3w10_linasnsga2_acc_macs.csv \
    --search_tactic evolutionary \
    --num_evals 250 \
    --search_algo nsga2
```

*Example 2b.* NAS process for the OFA ResNet50 super-network that optimizes for ImageNet Top-1 accuracy *and* model size (parameters) using a evolutionary AGE-MOEA approach.

```bash
python run_search.py \
    --supernet ofa_resnet50 \
    --optimization_metrics accuracy_top1 params \
    --measurements accuracy_top1 macs params \
    --results_path resnet50_age_acc_params.csv \
    --search_tactic evolutionary \
    --num_evals 500 \
    --search_algo age
```

### Many-Objective

*Example 3a.* NAS process for the OFA ResNet50 super-network that optimizes for ImageNet Top-1 accuracy *and* model size (parameters) *and* multiply-and-accumulates (MACs) using a evolutionary unsga3 approach.

```bash
python run_search.py \
    --supernet ofa_resnet50 \
    --optimization_metrics accuracy_top1 macs params \
    --measurements accuracy_top1 macs params \
    --results_path resnet50_linasunsga3_acc_macs_params.csv \
    --search_tactic evolutionary \
    --num_evals 500 \
    --search_algo unsga3
```

*Example 3b.* NAS process for the OFA MobileNetV3-w1.0 super-network that optimizes for ImageNet Top-1 accuracy *and* model size (parameters) *and* multiply-and-accumulates (MACs) using a linas+unsga3 approach.

```bash
python run_search.py \
    --supernet ofa_mbv3_d234_e346_k357_w1.0 \
    --optimization_metrics accuracy_top1 macs params \
    --measurements accuracy_top1 macs params \
    --results_path mbnv3w10_linasunsga3_acc_macs_params.csv \
    --search_tactic linas \
    --num_evals 500 \
    --search_algo unsga3
```

An example of the search results for a Multi-Objective search using both LINAS+NSGA-II and standard NSGA-II algorithms will yield results in the following format.
![DyNAS-T Results](https://github.com/IntelLabs/DyNAS-T/blob/main/docs/images/search_results.png?raw=true)

### Distributed Search

Search can be performed with multiple workers using the `MPI` / `torch.distributed` library. To use this functionality, your script should be called with `mpirun`/`mpiexec` command and an additional `--distributed` param has to be set (`DyNAS([...], distributed=True`).

> Note: When run with `torchrun`, unless explicitly specified, `torch.distributed` uses `OMP_NUM_THREADS=1` ([link](https://github.com/pytorch/pytorch/commit/1c0309a9a924e34803bf7e8975f7ce88fb845131)) which may result in slow evaluation time. Good practice is to explicitly set `OMP_NUM_THREADS`  to `(total_core_count)/(num_workers)` (optional for MPI).

*Example 4.* Distributed NAS process with two OpenMPI workers for the OFA MobileNetV3-w1.0 super-network that optimizes for ImageNet Top-1 accuracy *and* model size (parameters)

```bash
OMP_NUM_THREADS=28 mpirun \
    --report-bindings \
    -x MASTER_ADDR=127.0.0.1 \
    -x MASTER_PORT=1234 \
    -np 2 \
    -bind-to socket \
    -map-by socket \
    python run_search.py \
        --supernet ofa_mbv3_d234_e346_k357_w1.0 \
         --optimization_metrics accuracy_top1 macs \
        --results_path results.csv \
        --search_tactic linas \
        --distributed \
        --population 50 \
        --num_evals 250
```

## References

[1] Cai, H., Gan, C., & Han, S. (2020). Once for All: Train One Network and Specialize it for Efficient Deployment. ArXiv, abs/1908.09791.

[2] K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist multiobjective genetic algorithm: NSGA-II," in IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp. 182-197, April 2002, doi: 10.1109/4235.996017.

[3] Cummings, D., Sarah, A., Sridhar, S.N., Szankin, M., Muñoz, J.P., & Sundaresan, S. (2022). A Hardware-Aware Framework for Accelerating Neural Architecture Search Across Modalities. ArXiv, abs/2205.10358.

## Legal Disclaimer and Notices

> This “research quality code”  is for Non-Commercial purposes provided by Intel “As Is” without any express or implied warranty of any kind. Please see the dataset's applicable license for terms and conditions. Intel does not own the rights to this data set and does not confer any rights to it. Intel does not warrant or assume responsibility for the accuracy or completeness of any information, text, graphics, links or other items within the code. A thorough security review has not been performed on this code. Additionally, this repository may contain components that are out of date or contain known security vulnerabilities.

> ImageNet, WMT, SST2: Please see the dataset's applicable license for terms and conditions. Intel does not own the rights to this data set and does not confer any rights to it.
