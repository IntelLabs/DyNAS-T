import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from bnas_resnets import BootstrapNASResnet50
import time
import csv
import numpy as np
import autograd.numpy as anp
from datetime import datetime
import argparse
import copy
from fvcore.nn import FlopCountAnalysis, parameter_count

# DyNAS-T Specific Imports
from dynast.manager import ParameterManager
from dynast.evaluation_module.predictor import MobileNetAccuracyPredictor, MobileNetMACsPredictor
from dynast.search_module.search import SearchAlgoManager, ProblemMultiObjective
from dynast.analytics_module.results import ResultsManager


class BNASRunner:

    def __init__(self, supernet, acc_predictor, macs_predictor, val_loader, train_loader):
        self.supernet = supernet
        self.acc_predictor = acc_predictor
        self.macs_predictor = macs_predictor
        self.val_loader = val_loader
        self.train_loader = train_loader

    def validate_subnet(self, subnet_cfg):
        self.supernet.set_active_subnet(d=subnet_cfg['d'], e=subnet_cfg['e'], w=subnet_cfg['w'])
        subnet = self.supernet.get_active_subnet()
        self.reset_bn(subnet, 4000, 64)
        top1, top5, gflops = self.validate(subnet)
        model_params = self.count_parameters(subnet)

        return top1, top5, gflops, model_params

    def estimate_accuracy_top1(self, subnet_cfg):

        # Ridge Predictor
        top1 = self.acc_predictor.predict_single(subnet_cfg)
        return top1

    def estimate_macs(self, subnet_cfg):

        # Ridge Predictor
        macs = self.macs_predictor.predict_single(subnet_cfg)
        return macs

    def validate(self, model):
        model.to('cuda')
        model.eval()

        batch_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        gflops = 0

        len_val_loader = len(self.val_loader)

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.val_loader):
                images, labels = images.cuda(), labels.cuda()

                # Note, smaller batches do not save much time on FLOP measurement
                if i == 0:
                    start_time = time.time()
                    print('Running fvcore FLOP counter:')
                    flops = FlopCountAnalysis(model, images)
                    flop_batch_size = 64
                    gflops = flops.total()/(flop_batch_size*10**9)
                    print('GFLOPs: {}'.format(gflops))

                output = model(images)
                acc1, acc5 = self.accuracy(output, labels, topk=(1, 5))
                top1.update(acc1, images.size(0))
                top5.update(acc5, images.size(0))

                if i % 100 == 0:
                    print(
                        '{rank}'
                        'Val: [{0}/{1}] '
                        'Acc@1: {top1.val:.3f} ({top1.avg:.3f}) '
                        'Acc@5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                            i, len_val_loader,
                            top1=top1, top5=top5,
                            rank=''))
            print(
                'Val: '
                'Acc@1: {top1.val:.3f} ({top1.avg:.3f}) '
                'Acc@5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len_val_loader,
                    top1=top1, top5=top5))
            # if is_main_process():
            #     TODO: Tensorboard.
        return top1.avg, top5.avg, gflops


    def count_parameters(self, model):
        device = 'cpu'
        model.to(device)
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def reset_bn(self, model, num_samples, batch_size):
        model.train()
        if num_samples / batch_size > len(self.train_loader):
            print("BN set stats: num of samples exceed the samples in loader. Using full loader")
        for i, (images, _) in enumerate(self.train_loader):
            images = images.cuda()
            model(images)
            if i > num_samples / batch_size:
                print(f"Finishing setting bn stats using {num_samples} and batch size of {batch_size}")
                break

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size).item())
            return res


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class UserEvaluationInterface:
    '''
    The interface class update is required to be updated for each unique SuperNetwork
    framework as it controls how evaluation calls are made from DyNAS-T

    Parameters
    ----------
    evaluator : class
        The 'runner' that performs the validation or prediction
    manager : class
        The DyNAS-T manager that translates between PyMoo and the parameter dict
    csv_path : string
        (Optional) The csv file that get written to during the subnetwork search
    '''

    def __init__(self, evaluator, manager):
        self.evaluator = evaluator
        self.manager = manager

    def eval_subnet(self, x, validation=False, csv_path=None):
        # PyMoo vector to Elastic Parameter Mapping
        param_dict = self.manager.translate2param(x)

        sample = {
            'd': param_dict['d'],
            'e': param_dict['e'],
            'w': param_dict['w'],
        }

        # Bug Fix - deep copy prevents accidental re-mapping of sample
        subnet_sample = copy.deepcopy(sample)

        if validation:
            top1, top5, gflops, model_params  = self.evaluator.validate_subnet(subnet_sample)
        else:
            top1 = self.evaluator.estimate_accuracy_top1(self.manager.onehot_generic(x))
            gflops = self.evaluator.estimate_macs(self.manager.onehot_generic(x))


        if csv_path:
            with open(csv_path, 'a') as f:
                writer = csv.writer(f)
                date = str(datetime.now())
                result = [subnet_sample, date, float(gflops), float(top1)]
                writer.writerow(result)

        # PyMoo only minimizes objectives, thus accuracy needs to be negative
        # Requires format: subnetwork, objective x, objective y
        return sample, gflops, -top1



def main(args):

    #--------------------------------------
    # BootstramNAS Initialization
    #--------------------------------------

    supernet = BootstrapNASResnet50(depth_list=[0, 1],
                                    expand_ratio_list=[0.2, 0.25],
                                    width_mult_list=[0.65,0.8,1.0])

    print('Loading Model: supernet/torchvision_resnet50_supernet.pth')
    init = torch.load('supernet/torchvision_resnet50_supernet.pth')['state_dict']
    supernet.load_state_dict(init)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    print(f'[Info] Using device: {device}')
    if args.describe_models:
        print(summary(supernet.to(device),input_size=(3,224,224)))  # TODO(Maciej) Summary is never imported

    data_dir = '/datasets/imagenet-ilsvrc2012'
    valdir = data_dir + '/val'
    image_size = 224
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(int(image_size / 0.875)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]))
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, sampler=val_sampler, drop_last=True)

    traindir = data_dir + '/train'
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=64, shuffle=(train_sampler is None),
                num_workers=4, pin_memory=True, sampler=train_sampler, drop_last=True)  # , sampler=train_sampler)

    # --------------------------------
    # DyNAS-T Search Setup
    # --------------------------------

    # Define SuperNetwork Parameter Dictionary and Instantiate Manager
    supernet_parameters = {'d' : {'count' : 5,  'vars' : [0, 1]},
                           'e' : {'count' : 12, 'vars' : [0.2, 0.25]},
                           'w' : {'count' : 6,  'vars' : [0, 1, 2]} }
    supernet_manager = ParameterManager(param_dict=supernet_parameters,
                                        seed=args.seed)

    # Instatiate objective 'runner', treating latency LUT as ground truth for latency in this example
    runner = BNASRunner(supernet=supernet,
                            acc_predictor=None,
                            macs_predictor=None,
                            val_loader=val_loader,
                            train_loader=train_loader)

    # Define how evaluations occur, gives option for csv file
    validation_interface = UserEvaluationInterface(evaluator=runner,
                                                   manager=supernet_manager)

    # Concurrent Search
    validated_population = args.csv_path_val_output
    print(f'[Info] Validated population file: {args.csv_path_val_output}')

    # clear validation file
    with open(validated_population, 'w') as f:
        writer = csv.writer(f)
    last_population = [supernet_manager.random_sample() for _ in range(args.population)]
    #shutil.copy2(args.csv_path_initial_sample, val_file)

    # --------------------------------
    # DyNAS-T ConcurrentNAS Loop
    # --------------------------------

    num_loops = 10
    for loop in range(1, num_loops+1):
        print(f'[Info] Starting ConcurrentNAS loop {loop} of {num_loops}.')

        for individual in last_population:
            print(individual)
            validation_interface.eval_subnet(individual, validation=True, csv_path=validated_population)

        print('[Info] Training "weak" MACs predictor.')
        df = supernet_manager.import_csv(validated_population, config='config', objective='latency',
            column_names=['config','date','latency','top1'])
        features, labels = supernet_manager.create_training_set(df)
        macs_pred = MobileNetMACsPredictor()
        macs_pred.train(features, labels)

        print('[Info] Training "weak" accuracy predictor.')
        df = supernet_manager.import_csv(validated_population, config='config', objective='top1',
            column_names=['config','date','latency','top1'])
        features, labels = supernet_manager.create_training_set(df)
        acc_pred = MobileNetAccuracyPredictor()
        acc_pred.train(features, labels)

        runner_predictor = BNASRunner(supernet=supernet,
                                    acc_predictor=acc_pred,
                                    macs_predictor=macs_pred,
                                    val_loader=val_loader,
                                    train_loader=train_loader)

        prediction_interface = UserEvaluationInterface(evaluator=runner_predictor,
                                                       manager=supernet_manager)

        # Instantiate Multi-Objective Problem Class
        problem = ProblemMultiObjective(evaluation_interface=prediction_interface,
                                        param_count=supernet_manager.param_count,
                                        param_upperbound=supernet_manager.param_upperbound)

        # Instantiate Search Manager
        search_manager = SearchAlgoManager(algorithm=args.algorithm,
                                        seed=args.seed)
        search_manager.configure_nsga2(population=args.population,
                                       num_evals=100000)

        # Run the search!
        output = search_manager.run_search(problem)
        last_population = output.pop.get('X')

        # Process results
        results = ResultsManager(csv_path=f'./results_temp/loop{loop}_{args.csv_path}',
                                manager=supernet_manager,
                                search_output=output)
        results.history_to_csv()
        results.front_to_csv()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed',
        default=0, type=int, help='random seed')
    parser.add_argument('--describe_models',
        action='store_true', help='Train Predictors')
    parser.add_argument('--csv_path_val_output',
        required=True, default=None, help='location to save results.')
    parser.add_argument('--csv_path',
        required=False, default=None, help='location to save results.')
    parser.add_argument('--verbose',
        action='store_true', help='Flag to control output')
    parser.add_argument('--algorithm',
        default='nsga2', choices=['nsga2','rnsga2'],
        help='GA algorithm, currently supports nsga2 and rnsga2')
    parser.add_argument('--num_evals',
        default=100000, type=int, help='number of evaluations')
    parser.add_argument('--population',
        default=50, type=int, help='population size for each generation')

    args = parser.parse_args()

    print('[Info] Starting BootstrapNASResnet50 Search')
    main(args)








