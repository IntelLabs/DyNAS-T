# Reference Examples for running DyNAS-T

This section contains examples (or templates) that show how to integrate DyNAS-T search with various supernetwork platforms such in the HANDI framework.
Terminology:
* Full = subnetwork search using pre-trained predictors
* Warm = subnetwork search using pre-trained predictors starting from a 'warm' population
* Random = Randomly search the elastic paramenter space (subpar results)
* Concurrent = ConcurrentNAS approach to subnetwork search
* mbnv3 = MobileNetV3 Supernetwork

## HANDI MobileNetV3

* `dynast_mbnv3_full.py` - Full search example
* `dynast_mbnv3_concurrent.py` - Concurrent search example
* `dynast_mbnv3_concurrent_ov_int8.py` - Concurrent search with OpenVINO FP32->INT8 quantization space

## HANDI ResNet50

* `dynast_resnet50_full.py` - Full search example

## HANDI Transformer

* `dynast_transformer_full.py` - Full search example

## BootstrapNAS

* `dynast_bnas-resnet50_concurrent.py` - Concurrent search example for ResNet50 torchvision model
