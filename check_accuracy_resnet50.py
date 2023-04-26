import argparse
import copy
import sys

import torch
import yaml

# torch.set_num_threads(14)
# torch.set_num_interop_threads(2)

import os
import time

import torchvision

# from neural_compressor.experimental import Quantization, common
from neural_compressor import PostTrainingQuantConfig, quantization
from neural_compressor.utils.pytorch import load
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.quantization import ResNet50_QuantizedWeights
from torchvision.models.quantization import resnet50 as resnet50_quant

import dynast
import dynast.supernetwork.image_classification.ofa.ofa as ofa
from dynast.supernetwork.image_classification.ofa.ofa.imagenet_classification.elastic_nn.utils import (
    set_running_statistics,
)
from dynast.supernetwork.image_classification.ofa.ofa.model_zoo import ofa_net
from dynast.utils.datasets import Dataset
from dynast.utils.nn import measure_latency, reset_bn, validate_classification
from dynast.utils.reference import TorchVisionReference


def dynast_quantize(model_fp, qconfig_dict, data_loader=None, num_samples=None, log_out=False):
    model_fp.eval()

    model_qt = quantization.fit(model_fp, qconfig_dict, calib_dataloader=data_loader)

    return model_qt


def replace_relu(module):
    reassign = {}
    for name, mod in module.named_children():
        replace_relu(mod)
        # Checking for explicit type instead of instance
        # as we only want to replace modules of the exact type
        # not inherited classes
        if type(mod) == torch.nn.ReLU or type(mod) == torch.nn.ReLU6:
            reassign[name] = torch.nn.ReLU(inplace=False)

    for key, value in reassign.items():
        module._modules[key] = value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", type=str, default="resnet50")
    parser.add_argument(
        "--supernet",
        default="ofa_resnet50",
        choices=["ofa_mbv3_d234_e346_k357_w1.0", "ofa_mbv3_d234_e346_k357_w1.2", "ofa_resnet50"],
    )
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument(
        "-t", "--test_size", type=int, default=None, help="How many batches should be used for validation."
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="How many batches should be used to warm up latency measurement when benchmarking.",
    )
    parser.add_argument(
        "--measure_steps",
        type=int,
        default=50,
        help="How many batches should be used for actual latency measurement when benchmarking.",
    )
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dataset", type=str, choices=["imagenet", "imagenette", "cifar10"], default="imagenet")
    parser.add_argument("--dataset_dir", type=str, default="/datasets/imagenet-ilsvrc2012/")
    parser.add_argument("--calibration_samples", type=int, default=1000)
    parser.add_argument("--input_size", type=int, default=224)

    args = parser.parse_args()

    if args.model == "resnet50":
        reference_model_fp32 = resnet50(weights=ResNet50_Weights.DEFAULT)
        reference_model_int8 = resnet50_quant(weights=ResNet50_QuantizedWeights.IMAGENET1K_FBGEMM_V1, quantize=True)
    else:
        raise NotImplementedError(f"Unknown model: {args.model}")

    dataset = Dataset.get(args.dataset)
    dataset.PATH = args.dataset_dir
    val_dataloader = dataset.validation_dataloader(args.batch_size)

    if False:
        _, top1_fp32, top5_fp32 = validate_classification(
            model=reference_model_fp32,
            data_loader=val_dataloader,
        )
    else:
        # Finished validate_classification in 1064.9178 s
        top1_fp32 = 76.15637003841229
        top5_fp32 = 92.87972151088348
    print(f"{top1_fp32=} {top5_fp32=}")

    if False:
        _, top1_int8, top5_int8 = validate_classification(
            model=reference_model_int8,
            data_loader=val_dataloader,
        )
    else:
        # Finished validate_classification in 149.2050 s
        # top1_int8=75.98758012820512 top5_int8=92.8485576923077
        top1_int8 = 75.98758012820512
        top5_int8 = 92.8485576923077
    print(f"{top1_int8=} {top5_int8=}")

    if args.supernet == "ofa_resnet50":
        supernet = ofa_net(args.supernet, pretrained=True)
        supernet.set_active_subnet(
            d=[1] * 5,  # , 3, 3, 5, 3, 7, 7, 7, 5, 3, 5, 7, 7, 7, 5, 7, 5, 5, 5, 3],
            e=[0.2] * 18,  # [0.25]*18,#[6, 4, 6, 6, 6, 4, 4, 4, 4, 3, 6, 4, 3, 3, 3, 4, 3, 3, 3, 4],
            w=[0] * 6,
        ),  # [3, 2, 3, 2, 2],
        subnet_fp32 = supernet.get_active_subnet(preserve_weight=True)
    else:
        print(f"Undefined SuperNet: {args.supernet}! -- Using Torch Reference Model: {args.model}")

    # replace_relu(ref.model)

    print("=" * 40, " quantized model ", "=" * 40)

    # ref.dataset.sub_train_dataloader(batch_size=64, image_size=args.input_size,n_images=2000,train_dir=os.path.join(args.dataset_dir,"train"))
    # sub_train_loader = ref.dataset.validation_dataloader(batch_size=200, image_size=args.input_size, shuffle=False, val_dir=os.path.join(args.dataset_dir, "val"))

    train_dataloader_full = dataset.train_dataloader(args.batch_size, shuffle=False)
    # subset_train_loader =  torch.utils.data.Subset(train_dataloader_full, list(range(args.calibration_samples)))
    # train_dataloader = DataLoader(subset_train_loader, shuffle=False, batch_size=args.batch_size)

    if False or not os.path.exists("subnet.pt"):
        set_running_statistics(subnet_fp32, train_dataloader_full, calibration_samples=args.calibration_samples)
        torch.save(subnet_fp32, "subnet.pt")
        print("saving subnet.")
    else:
        print("loading subnet.")
        subnet_fp32 = torch.load("subnet.pt")
    subnet_fp32.eval()

    if False:
        _, top1_subfp32, top5_subfp32 = validate_classification(
            model=subnet_fp32,
            data_loader=val_dataloader,
        )
    else:
        top1_subfp32 = 69.58934294871794
        top5_subfp32 = 88.90024038461539
    print(f"{top1_subfp32=} {top5_subfp32=}")

    if False or not os.path.exists("subnet_int8"):
        qconfig_dict = PostTrainingQuantConfig(
            backend="default",
            calibration_sampling_size=[args.calibration_samples],
        )
        # approach="static",calibration_sampling_size=[args.calibration_samples]),
        # op_name_list={"first_conv.conv":  {'weight': {'dtype':['fp32']},'activation': {'dtype':['fp32']}},
        #              #"first_conv.act":  {'weight': {'dtype':['fp32']},'activation': {'dtype':['fp32']}},
        #              "blocks.0.conv.point_linear.conv":  {'weight': {'dtype':['fp32']},'activation': {'dtype':['fp32']}},
        #              "final_expand_layer.conv":  {'weight': {'dtype':['fp32']},'activation': {'dtype':['fp32']}},
        #              "feature_mix_layer.conv": {'weight': {'dtype':['fp32']},'activation': {'dtype':['fp32']}},
        #               "classifier.linear":{'weight': {'dtype':['fp32']},'activation': {'dtype':['fp32']}} },)
        # "blocks.0.conv.depth_conv.conv":  {'weight': {'dtype':['fp32']},'activation': {'dtype':['fp32']}},
        # "blocks.0.conv.depth_conv.bn":  {'weight': {'dtype':['fp32']},'activation': {'dtype':['fp32']}},
        # "blocks.0.conv.depth_conv.act":  {'weight': {'dtype':['fp32']},'activation': {'dtype':['fp32']}}})
        calib_batch_size = args.batch_size
        # calibrate_dataloader = ref.dataset.validation_dataloader(batch_size=calib_batch_size, image_size=args.input_size,val_dir=os.path.join(args.dataset_dir, "val"))
        # calibrate_dataloader = ref.dataset.train_dataloader(batch_size=64, image_size=args.input_size,train_dir=os.path.join(args.dataset_dir,"train")))
        subnet_int8 = dynast_quantize(
            subnet_fp32, qconfig_dict, val_dataloader, args.calibration_samples, log_out=False
        )
        # torch.save(subnet_int8, 'subnet_q.pt')
        subnet_int8.save("subnet_int8")
        print("saving subnet_int8.")

        _, top1_sub_int8, top5_sub_int8 = validate_classification(
            model=subnet_int8,
            data_loader=val_dataloader,
        )
    else:
        print("loading subnet_int8.")
        subnet_int8 = load(
            os.path.abspath(os.path.expanduser("subnet_int8")),
            subnet_fp32,
            dataloader=val_dataloader,
        )
        top1_sub_int8 = 68.7119391025641
        top5_sub_int8 = 88.3994391025641
    print(f"{top1_sub_int8=} {top5_sub_int8=}")

    latency_mean, latency_std = measure_latency(subnet_int8, (args.batch_size, 3, 244, 244))
    print(f"lat {latency_mean} {latency_std}")
