# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import warnings

from .utils import *

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser('BEiT fine-tuning and evaluation script for image classification', add_help=False)

    # Model parameters
    parser.add_argument(
        '--model', default='beit_base_patch16_224', type=str, metavar='MODEL', help='Name of model to train'
    )
    parser.add_argument(
        '--task',
        type=str,
        required=False,
        choices=['nlvr2', 'vqav2', 'flickr30k', 'coco_retrieval', 'coco_captioning', 'nocaps', 'imagenet'],
        help='Name of task to fine-tuning',
    )

    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')

    parser.add_argument(
        '--checkpoint_activations', action='store_true', default=None, help='Enable checkpointing to save your memory.'
    )
    parser.add_argument(
        '--sentencepiece_model', type=str, required=False, help='Sentencepiece model path for the pretrained model.'
    )
    parser.add_argument('--vocab_size', type=int, default=64010)
    parser.add_argument('--num_max_bpe_tokens', type=int, default=64)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    parser.add_argument(
        '--opt_eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)'
    )
    parser.add_argument(
        '--opt_betas',
        default=[0.9, 0.999],
        type=float,
        nargs='+',
        metavar='BETA',
        help='Optimizer Betas (default: 0.9, 0.999, use opt default)',
    )
    parser.add_argument(
        '--clip_grad', type=float, default=None, metavar='NORM', help='Clip gradient norm (default: None, no clipping)'
    )
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR', help='learning rate (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=0.9)
    parser.add_argument('--task_head_lr_weight', type=float, default=0)

    parser.add_argument(
        '--warmup_lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)'
    )
    parser.add_argument(
        '--min_lr',
        type=float,
        default=1e-6,
        metavar='LR',
        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)',
    )
    parser.add_argument(
        '--warmup_epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports'
    )
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=-1,
        metavar='N',
        help='num of steps to warmup LR, will overload warmup_epochs if set > 0',
    )

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--eval_batch_size', default=None, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

    # Augmentation parameters
    parser.add_argument('--randaug', action='store_true', default=False)
    parser.add_argument(
        '--train_interpolation',
        type=str,
        default='bicubic',
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")',
    )

    # Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str, help='dataset path')

    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None, help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument(
        '--pin_mem',
        action='store_true',
        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.',
    )
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # parameter for dump predictions (VQA, COCO captioning, NoCaps)
    parser.add_argument('--task_cache_path', default=None, type=str)

    # parameter for imagenet finetuning
    parser.add_argument('--nb_classes', default=1000, type=int, help='number of the classification types')
    parser.add_argument('--mixup', type=float, default=0, help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0, help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument(
        '--cutmix_minmax',
        type=float,
        nargs='+',
        default=None,
        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)',
    )
    parser.add_argument(
        '--mixup_prob',
        type=float,
        default=1.0,
        help='Probability of performing mixup or cutmix when either/both is enabled',
    )
    parser.add_argument(
        '--mixup_switch_prob',
        type=float,
        default=0.5,
        help='Probability of switching to cutmix when both mixup and cutmix enabled',
    )
    parser.add_argument(
        '--mixup_mode',
        type=str,
        default='batch',
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # augmentation parameters for imagenet finetuning
    parser.add_argument(
        '--color_jitter', type=float, default=0.4, metavar='PCT', help='Color jitter factor (default: 0.4)'
    )
    parser.add_argument(
        '--aa',
        type=str,
        default='rand-m9-mstd0.5-inc1',
        metavar='NAME',
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)',
    )
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

    # evaluation parameters for imagenet
    parser.add_argument('--crop_pct', type=float, default=None)

    # random Erase params for imagenet finetuning
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT', help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')
    parser.add_argument(
        '--resplit', action='store_true', default=False, help='Do not random erase first (clean) augmentation split'
    )

    # parameter for captioning finetuning
    parser.add_argument('--captioning_mask_prob', type=float, default=0.6)
    parser.add_argument('--drop_worst_ratio', type=float, default=0.2)
    parser.add_argument('--drop_worst_after', type=int, default=12000)
    parser.add_argument('--num_beams', type=int, default=3)
    parser.add_argument('--length_penalty', type=float, default=0.6)

    # label smoothing for imagenet and captioning
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    # deepspeed parameters
    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--initial_scale_power', type=int, default=16)
    parser.add_argument('--zero_stage', default=0, type=int, help='ZeRO optimizer stage (default: 0)')

    parser.add_argument(
        '--supernet',
        default='ofa_mbv3_d234_e346_k357_w1.0',
        type=str,
        help='Super-network',
        choices=[
            'ofa_resnet50',
            'ofa_mbv3_d234_e346_k357_w1.0',
            'ofa_mbv3_d234_e346_k357_w1.2',
            'ofa_proxyless_d234_e346_k357_w1.3',
            'transformer_lt_wmt_en_de',
            'bert_base_sst2',
            'beit3_imagenet',
        ],
    )
    # TODO(macsz) might be better to list allowed elements here (or in help message)
    parser.add_argument(
        '--optimization_metrics',
        default=['accuracy_top1', 'macs'],  # TODO*macsz) Consider just 'accuracy',
        # as it translates better for both image classification
        # and Transformer LT
        nargs='*',
        type=str,
        help='Metrics that will be optimized for during search.',
    )
    parser.add_argument(
        '--measurements',
        default=['accuracy_top1', 'macs', 'params', 'latency'],  # TODO(macsz) Ditto
        nargs='*',
        type=str,
        help='Measurements during search.',
    )
    parser.add_argument('--num_evals', default=250, type=int, help='Total number of evaluations during search.')

    parser.add_argument(
        '--valid_size',
        default=None,
        type=int,
        help='How many batches of data to use when evaluating model\'s accuracy.',
    )
    parser.add_argument('--dataloader_workers', default=4, type=int, help='How many workers to use when loading data.')
    parser.add_argument('--population', default=50, type=int, help='Population size for each generation')
    parser.add_argument('--results_path', required=True, type=str, help='Path to store search results, csv format')
    parser.add_argument(
        '--dataset_path', default='/panfs/projects/ML_datasets/imagenet/ilsvrc12_raw/', type=str, help=''
    )
    parser.add_argument('--supernet_ckpt_path', default=None, type=str, help='Path to supernet checkpoint.')
    parser.add_argument(
        '--test_fraction',
        default=1.0,
        type=float,
        help='Fraction of the validation data to be used for testing and evaluation.',
    )
    parser.add_argument(
        '--search_tactic',
        default='linas',
        choices=['linas', 'evolutionary', 'random'],
        help='Search tactic, currently supports: ["linas", "evolutionary" , "random"]',
    )
    parser.add_argument(
        '--search_algo',
        default='nsga2',
        choices=['cmaes', 'nsga2', 'age', 'unsga3', 'ga', 'ctaea', 'moead'],
        help='Search algorithm, currently supports: ["cmaes", "nsga2", "age", "unsga3", "ga", "ctaea", "moead"]',
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='Print more information.')

    dist_parser = parser.add_argument_group('Distributed search options')
    parser.add_argument(
        '--distributed',
        action='store_true',
        help='If set, a distributed implementation of the search algorithm will be used.',
    )

    dist_parser.add_argument("--backend", type=str, default="gloo", choices=['gloo'])
    parser.add_argument(
        '--mixed_precision',
        '--mp',
        action='store_true',
        help='If set, search will be performed over both architecture and mixed precision configurations.',
    )

    ds_init = None

    return parser.parse_args(), ds_init
