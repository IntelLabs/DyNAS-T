import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from bnas_resnets import BootstrapNASResnet50
from tqdm import tqdm

from dynast.quantization import quantize_ov, validate
from dynast.utils import (count_parameters, get_gflops, log, measure_time,
                          samples_to_batch_multiply)
from dynast.utils.ov import benchmark_openvino

SUPERNET_PARAMETERS = {
    'd': {'count': 5,  'vars': [0, 1]},
    'e': {'count': 12, 'vars': [0.2, 0.25]},
    'w': {'count': 6,  'vars': [0, 1, 2]},
}


class BNASRunner:

    def __init__(self,
                 supernet,
                 acc_predictor=None,
                 latency_predictor=None,
                 macs_predictor=None,
                 bn_samples=4000,
                 batch_size=128,
                 resolution=224,
                 cores=56
                 ):
        self.supernet = supernet
        self.acc_predictor = acc_predictor
        self.latency_predictor = latency_predictor
        self.macs_predictor = macs_predictor
        self.bn_samples = bn_samples
        self.batch_size = batch_size
        self.resolution = resolution
        self.cores = cores
        self.img_size = (self.batch_size, 3, self.resolution, self.resolution)

    def estimate_accuracy_top1(self, subnet_cfg):
        assert self.acc_predictor, 'Please provide `acc_predictor` when creating BNASRunner object.'
        top1 = self.acc_predictor.predict_single(subnet_cfg)
        return top1

    def estimate_latency(self, subnet_cfg):
        assert self.acc_predictor, 'Please provide `latency_predictor` when creating BNASRunner object.'
        lat = self.latency_predictor.predict_single(subnet_cfg)
        return lat

    def validate_subnet(self, subnet_cfg):
        top1, top5 = validate()

        self.supernet.set_active_subnet(d=subnet_cfg['d'], e=subnet_cfg['e'], w=subnet_cfg['w'])
        subnet = self.supernet.get_active_subnet()
        gflops = get_gflops(subnet)
        model_params = count_parameters(subnet)

        return top1, top5, gflops, model_params

    def benchmark_subnet(self):
        latency_ov, fps_ov = benchmark_openvino(
            self.img_size,
            cores=self.cores,
            is_quantized=True,
        )

        return latency_ov

    def quantize(self, subnet_cfg):
        self.supernet.set_active_subnet(d=subnet_cfg['d'], e=subnet_cfg['e'], w=subnet_cfg['w'])
        subnet = self.supernet.get_active_subnet()

        reset_bn(
            subnet,
            samples_to_batch_multiply(self.bn_samples, self.batch_size),
            self.batch_size,
        )

        quantize_ov(
            model=subnet,
            img_size=self.img_size
        )


def get_supernet(path='supernet/torchvision_resnet50_supernet.pth', device='cpu'):
    supernet = BootstrapNASResnet50(depth_list=[0, 1], expand_ratio_list=[
                                    0.2, 0.25], width_mult_list=[0.65, 0.8, 1.0])

    init = torch.load(path, map_location=torch.device(device))['state_dict']
    supernet.load_state_dict(init)
    return supernet


@measure_time
def reset_bn(model, num_samples, batch_size, device='cpu'):
    train_loader = get_train_loader()
    model.train()
    if num_samples / batch_size > len(train_loader):
        log.warn(
            "BN set stats: num of samples exceed the samples in loader. Using full loader")
    for i, (images, _) in tqdm(enumerate(train_loader), total=num_samples // batch_size, desc='Reset BN'):
        if 'cpu' not in device:
            images = images.cuda()
        model(images)
        if i > num_samples / batch_size:
            log.info(f"Finishing setting bn stats using {num_samples} and batch size of {batch_size}")
            break


def get_transform_normalize():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_train_loader(batch_size=128, image_size=224, train_dir='/datasets/imagenet-ilsvrc2012/train'):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        get_transform_normalize(),
    ])
    train_dataset = datasets.ImageFolder(
        train_dir,
        train_transforms,
    )
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=4, pin_memory=True, sampler=train_sampler, drop_last=True
    )

    return train_loader


def get_val_lovader(batch_size=128, image_size=224, val_dir='/datasets/imagenet-ilsvrc2012/train'):
    val_transforms = transforms.Compose([
        transforms.Resize(int(image_size / 0.875)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        get_transform_normalize(),
    ])
    val_dataset = datasets.ImageFolder(
        val_dir,
        val_transforms,
    )
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, sampler=val_sampler, drop_last=True
    )
    return val_loader
