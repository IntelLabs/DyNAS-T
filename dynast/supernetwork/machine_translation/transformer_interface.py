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

"""
Translate pre-processed data with a trained model.
"""
import copy
import csv
import ctypes
import logging
import math
import os
import sys
import time
import warnings
from datetime import datetime
from typing import Tuple

import fairseq
import numpy as np
import torch
import torchprofile
from fairseq.data import dictionary

from dynast.search.evaluation_interface import EvaluationInterface
from dynast.utils import log

from .transformer_supernetwork import TransformerSuperNetwork

warnings.filterwarnings("ignore")


try:
    from fairseq import libbleu
except ImportError as e:
    import sys

    sys.stderr.write('ERROR: missing libbleu.so. run `pip install --editable .`\n')
    raise e


C = ctypes.cdll.LoadLibrary(libbleu.__file__)


class BleuStat(ctypes.Structure):
    _fields_ = [
        ('reflen', ctypes.c_size_t),
        ('predlen', ctypes.c_size_t),
        ('match1', ctypes.c_size_t),
        ('count1', ctypes.c_size_t),
        ('match2', ctypes.c_size_t),
        ('count2', ctypes.c_size_t),
        ('match3', ctypes.c_size_t),
        ('count3', ctypes.c_size_t),
        ('match4', ctypes.c_size_t),
        ('count4', ctypes.c_size_t),
    ]


class Scorer(object):
    def __init__(self, pad, eos, unk):
        self.stat = BleuStat()
        self.pad = pad
        self.eos = eos
        self.unk = unk
        self.reset()

    def reset(self, one_init=False):
        if one_init:
            C.bleu_one_init(ctypes.byref(self.stat))
        else:
            C.bleu_zero_init(ctypes.byref(self.stat))

    def add(self, ref, pred):
        if not isinstance(ref, torch.IntTensor):
            raise TypeError('ref must be a torch.IntTensor (got {})'.format(type(ref)))
        if not isinstance(pred, torch.IntTensor):
            raise TypeError('pred must be a torch.IntTensor(got {})'.format(type(pred)))

        # don't match unknown words
        rref = ref.clone()
        assert not rref.lt(0).any()
        rref[rref.eq(self.unk)] = -999

        rref = rref.contiguous().view(-1)
        pred = pred.contiguous().view(-1)

        C.bleu_add(
            ctypes.byref(self.stat),
            ctypes.c_size_t(rref.size(0)),
            ctypes.c_void_p(rref.data_ptr()),
            ctypes.c_size_t(pred.size(0)),
            ctypes.c_void_p(pred.data_ptr()),
            ctypes.c_int(self.pad),
            ctypes.c_int(self.eos),
        )

    def score(self, order=4):
        psum = sum(math.log(p) if p > 0 else float('-Inf') for p in self.precision()[:order])
        return self.brevity() * math.exp(psum / order) * 100

    def precision(self):
        def ratio(a, b):
            return a / b if b > 0 else 0

        return [
            ratio(self.stat.match1, self.stat.count1),
            ratio(self.stat.match2, self.stat.count2),
            ratio(self.stat.match3, self.stat.count3),
            ratio(self.stat.match4, self.stat.count4),
        ]

    def brevity(self):
        r = self.stat.reflen / self.stat.predlen
        return min(1, math.exp(1 - r))

    def result_string(self, order=4):
        assert order <= 4, "BLEU scores for order > 4 aren't supported"
        fmt = 'BLEU{} = {:2.2f}, {:2.1f}'
        for _ in range(1, order):
            fmt += '/{:2.1f}'
        fmt += ' (BP={:.3f}, ratio={:.3f}, syslen={}, reflen={})'
        bleup = [p * 100 for p in self.precision()[:order]]
        return fmt.format(
            order,
            self.score(order=order),
            *bleup,
            self.brevity(),
            self.stat.predlen / self.stat.reflen,
            self.stat.predlen,
            self.stat.reflen,
        )


def compute_bleu(config, dataset_path, checkpoint_path):
    """Measure BLEU score of the Transformer-based model."""
    options = fairseq.options
    utils = fairseq.utils
    tasks = fairseq.tasks
    MosesTokenizer = fairseq.data.encoders.moses_tokenizer.MosesTokenizer
    StopwatchMeter = fairseq.meters.StopwatchMeter
    progress_bar = fairseq.progress_bar

    parser = options.get_generation_parser()

    args = options.parse_args_and_arch(parser, [dataset_path])

    args.data = dataset_path
    args.beam = 5
    args.remove_bpe = '@@ '
    args.gen_subset = 'test'
    args.lenpen = 0.6
    args.source_lang = 'en'
    args.target_lang = 'de'
    args.batch_size = 128
    args.eval_bleu_remove_bpe = '@@ '
    args.eval_bleu_detok = 'moses'

    utils.import_user_module(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # when running on CPU, use fp32 as default
    if not use_cuda:
        args.fp16 = False

    torch.manual_seed(args.seed)

    # Optimize ensemble for generation
    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    tokenizer = MosesTokenizer(args)
    task.tokenizer = tokenizer
    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    model = TransformerSuperNetwork(task)
    state = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    model.load_state_dict(state['model'], strict=True)

    if use_cuda:
        model.cuda()
    model.set_sample_config(config)
    model.make_generation_fast_(
        beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
        need_attn=args.print_alignment,
    )
    if args.fp16:
        model.half()
    if use_cuda:
        model.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=128,
        max_positions=utils.resolve_max_positions(task.max_positions(), *[model.max_positions()]),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator([model], args)

    num_sentences = 0
    bleu_list = []
    with progress_bar.build_progress_bar(args, itr) as t:
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                continue

            bleu = task._inference_with_bleu(generator, sample, model)
            bleu_list.append(bleu.score)

            num_sentences += sample['nsentences']

    bleu_score = np.mean(np.array(bleu_list))
    return bleu_score


def compute_latency(config, dataset_path, batch_size=128, get_model_parameters=False):
    """Measure latency of the Transformer-based model."""
    options = fairseq.options
    utils = fairseq.utils
    tasks = fairseq.tasks

    parser = options.get_generation_parser()

    args = options.parse_args_and_arch(parser, [dataset_path])

    args.data = dataset_path
    args.beam = 5
    args.remove_bpe = '@@ '
    args.gen_subset = 'test'
    args.lenpen = 0.6
    args.source_lang = 'en'
    args.target_lang = 'de'
    args.batch_size = batch_size
    utils.import_user_module(args)
    args.latgpu = False
    args.latcpu = True
    args.latiter = 100

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Optimize ensemble for generation
    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Load ensemble
    model = TransformerSuperNetwork(task)

    # specify the length of the dummy input for profile
    # for iwslt, the average length is 23, for wmt, that is 30
    dummy_sentence_length_dict = {'iwslt': 23, 'wmt': 30}

    dummy_sentence_length = dummy_sentence_length_dict['wmt']

    dummy_src_tokens = [2] + [7] * (dummy_sentence_length - 1)
    dummy_prev = [7] * (dummy_sentence_length - 1) + [2]

    src_tokens_test = torch.tensor([dummy_src_tokens], dtype=torch.long)
    src_lengths_test = torch.tensor([dummy_sentence_length])
    prev_output_tokens_test_with_beam = torch.tensor([dummy_prev] * args.beam, dtype=torch.long)
    bsz = 1
    new_order = torch.arange(bsz).view(-1, 1).repeat(1, args.beam).view(-1).long()
    if args.latcpu:
        model.cpu()
        log.info('Measuring model latency on CPU for dataset generation...')
    elif args.latgpu:
        model.cuda()
        src_tokens_test = src_tokens_test
        src_lengths_test = src_lengths_test
        prev_output_tokens_test_with_beam = prev_output_tokens_test_with_beam
        log.info('Measuring model latency on GPU for dataset generation...')
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

    model.set_sample_config(config)

    model.eval()

    with torch.no_grad():

        # dry runs
        for _ in range(15):
            encoder_out_test = model.encoder(src_tokens=src_tokens_test, src_lengths=src_lengths_test)

        encoder_latencies = []
        log.info('Measuring encoder for dataset generation...')
        for _ in range(args.latiter):
            if args.latgpu:
                start = time.time()
            elif args.latcpu:
                start = time.time()

            model.encoder(src_tokens=src_tokens_test, src_lengths=src_lengths_test)

            if args.latgpu:
                end = time.time()
                encoder_latencies.append((end - start) * 1000)
            elif args.latcpu:
                end = time.time()
                encoder_latencies.append((end - start) * 1000)

        encoder_latencies.sort()
        encoder_latencies = encoder_latencies[int(args.latiter * 0.1) : -max(1, int(args.latiter * 0.1))]
        log.info(
            f'Encoder latency for dataset generation: Mean: '
            '{np.mean(encoder_latencies)} ms; Std: {np.std(encoder_latencies)} ms'
        )

        encoder_out_test_with_beam = model.encoder.reorder_encoder_out(encoder_out_test, new_order)

        # dry runs
        for _ in range(15):
            model.decoder(prev_output_tokens=prev_output_tokens_test_with_beam, encoder_out=encoder_out_test_with_beam)

        # decoder is more complicated because we need to deal with incremental states and auto regressive things
        decoder_iterations_dict = {'iwslt': 23, 'wmt': 30}

        decoder_iterations = decoder_iterations_dict['wmt']
        decoder_latencies = []

        log.info('Measuring decoder for dataset generation...')
        for _ in range(args.latiter):
            if args.latgpu:
                start = time.time()
            elif args.latcpu:
                start = time.time()
            incre_states = {}
            for k_regressive in range(decoder_iterations):
                model.decoder(
                    prev_output_tokens=prev_output_tokens_test_with_beam[:, : k_regressive + 1],
                    encoder_out=encoder_out_test_with_beam,
                    incremental_state=incre_states,
                )
            if args.latgpu:
                end = time.time()
                decoder_latencies.append((end - start) * 1000)

            elif args.latcpu:
                end = time.time()
                decoder_latencies.append((end - start) * 1000)

        # only use the 10% to 90% latencies to avoid outliers
        decoder_latencies.sort()
        decoder_latencies = decoder_latencies[int(args.latiter * 0.1) : -max(1, int(args.latiter * 0.1))]

    log.info(
        f'Decoder latency for dataset generation: Mean: {np.mean(decoder_latencies)} ms; \t Std: {np.std(decoder_latencies)} ms'
    )

    lat_mean = np.mean(encoder_latencies) + np.mean(decoder_latencies)
    lat_std = np.std(encoder_latencies) + np.std(decoder_latencies)
    return lat_mean, lat_std


def compute_macs(config, dataset_path):
    """Calculate MACs for Transformer-based models."""
    options = fairseq.options
    utils = fairseq.utils
    tasks = fairseq.tasks

    parser = options.get_generation_parser()

    args = options.parse_args_and_arch(parser, [dataset_path])

    args.data = dataset_path
    args.beam = 5
    args.remove_bpe = '@@ '
    args.gen_subset = 'test'
    args.lenpen = 0.6
    args.source_lang = 'en'
    args.target_lang = 'de'
    args.batch_size = 128
    utils.import_user_module(args)
    args.latgpu = False
    args.latcpu = True
    args.latiter = 100

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Optimize ensemble for generation
    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Load model
    log.info('Loading model(s) from {}'.format(args.path))
    model = TransformerSuperNetwork(task)

    # specify the length of the dummy input for profile
    # for iwslt, the average length is 23, for wmt, that is 30
    dummy_sentence_length_dict = {'iwslt': 23, 'wmt': 30}

    dummy_sentence_length = dummy_sentence_length_dict['wmt']

    dummy_src_tokens = [2] + [7] * (dummy_sentence_length - 1)
    dummy_prev = [7] * (dummy_sentence_length - 1) + [2]

    model.eval()
    model.profile(mode=True)
    model.set_sample_config(config)
    macs = torchprofile.profile_macs(
        model,
        args=(
            torch.tensor([dummy_src_tokens], dtype=torch.long),
            torch.tensor([30]),
            torch.tensor([dummy_prev], dtype=torch.long),
        ),
    )

    model.profile(mode=False)

    params = model.get_sampled_params_numel(config)
    return macs, params


class TransformerLTRunner:
    """The OFARunner class manages the sub-network selection from the OFA super-network and
    the validation measurements of the sub-networks. ResNet50, MobileNetV3 w1.0, and MobileNetV3 w1.2
    are currently supported. Imagenet is required for these super-networks `imagenet-ilsvrc2012`.
    """

    def __init__(
        self,
        supernet,
        dataset_path,
        acc_predictor=None,
        macs_predictor=None,
        latency_predictor=None,
        params_predictor=None,
        batch_size=1,
        checkpoint_path=None,
    ):

        self.supernet = supernet
        self.acc_predictor = acc_predictor
        self.macs_predictor = macs_predictor
        self.latency_predictor = latency_predictor
        self.params_predictor = params_predictor
        self.batch_size = batch_size
        self.device = 'cpu'  # TODO(macsz) It's not used anywhere
        self.test_size = None
        self.dataset_path = dataset_path
        self.checkpoint_path = checkpoint_path

    def estimate_bleu(
        self,
        subnet_cfg: dict,
    ) -> float:
        top1 = self.acc_predictor.predict(subnet_cfg)
        return top1

    def estimate_macs(
        self,
        subnet_cfg: dict,
    ) -> int:
        macs = self.macs_predictor.predict(subnet_cfg)
        return macs

    def estimate_latency(
        self,
        subnet_cfg: dict,
    ) -> float:
        latency = self.latency_predictor.predict(subnet_cfg)
        return latency

    def validate_bleu(
        self,
        subnet_cfg: dict,
    ) -> float:  # pragma: no cover

        bleu = compute_bleu(subnet_cfg, self.dataset_path, self.checkpoint_path)
        return bleu

    def validate_macs(
        self,
        subnet_cfg: dict,
    ) -> Tuple[float, float]:
        """Measure Torch model's FLOPs/MACs as per FVCore calculation
        Args:
            subnet_cfg: sub-network Torch model
        Returns:
            `macs`
        """
        macs, params = compute_macs(subnet_cfg, self.dataset_path)
        logging.info('Model\'s macs: {}'.format(macs))
        return macs, params

    @torch.no_grad()
    def measure_latency(
        self,
        subnet_cfg: dict,
        input_size: tuple = (1, 3, 224, 224),
        warmup_steps: int = 10,
        measure_steps: int = 50,
        device: str = 'cpu',
    ):
        """Measure Torch model's latency.
        Args:
            subnet_cfg: sub-network Torch model
        Returns:
            mean latency; std latency
        """

        logging.info(
            f'Performing Latency measurements. Warmup = {warmup_steps},\
             Measure steps = {measure_steps}'
        )

        times = []
        lat_mean, lat_std = compute_latency(subnet_cfg, self.dataset_path)
        logging.info('Model\'s latency: {} +/- {}'.format(lat_mean, lat_std))

        return lat_mean, lat_std


class EvaluationInterfaceTransformerLT(EvaluationInterface):
    def __init__(
        self,
        evaluator,
        manager,
        optimization_metrics: list = ['bleu', 'latency'],
        measurements: list = ['bleu', 'latency'],
        csv_path=None,
        predictor_mode: bool = False,
    ):
        super().__init__(evaluator, manager, optimization_metrics, measurements, csv_path, predictor_mode)

    def eval_subnet(self, x):
        # PyMoo vector to Elastic Parameter Mapping
        param_dict = self.manager.translate2param(x)

        sample = {
            'encoder': {
                'encoder_embed_dim': param_dict['encoder_embed_dim'][0],
                'encoder_layer_num': 6,
                'encoder_ffn_embed_dim': param_dict['encoder_ffn_embed_dim'],
                'encoder_self_attention_heads': param_dict['encoder_self_attention_heads'],
            },
            'decoder': {
                'decoder_embed_dim': param_dict['decoder_embed_dim'][0],
                'decoder_layer_num': param_dict['decoder_layer_num'][0],
                'decoder_ffn_embed_dim': param_dict['decoder_ffn_embed_dim'],
                'decoder_self_attention_heads': param_dict['decoder_self_attention_heads'],
                'decoder_ende_attention_heads': param_dict['decoder_ende_attention_heads'],
                'decoder_arbitrary_ende_attn': param_dict['decoder_arbitrary_ende_attn'],
            },
        }
        subnet_sample = copy.deepcopy(sample)

        individual_results = dict()
        for metric in ['params', 'latency', 'macs', 'bleu']:
            individual_results[metric] = 0

        # Predictor Mode
        if self.predictor_mode == True:
            if 'params' in self.optimization_metrics:
                individual_results['params'] = self.evaluator.estimate_parameters(
                    self.manager.onehot_custom(param_dict).reshape(1, -1)
                )[0]
            if 'latency' in self.optimization_metrics:
                individual_results['latency'] = self.evaluator.estimate_latency(
                    self.manager.onehot_custom(param_dict).reshape(1, -1)
                )[0]
            if 'macs' in self.optimization_metrics:
                individual_results['macs'] = self.evaluator.estimate_macs(
                    self.manager.onehot_custom(param_dict).reshape(1, -1)
                )[0]
            if 'bleu' in self.optimization_metrics:
                individual_results['bleu'] = self.evaluator.estimate_bleu(
                    self.manager.onehot_custom(param_dict).reshape(1, -1)
                )[0]

        # Validation Mode
        else:
            if 'macs' in self.measurements or 'params' in self.measurements:
                individual_results['macs'], individual_results['params'] = self.evaluator.validate_macs(subnet_sample)
            if 'latency' in self.measurements:
                individual_results['latency'], _ = self.evaluator.measure_latency(subnet_sample)
            if 'bleu' in self.measurements:
                individual_results['bleu'] = self.evaluator.validate_bleu(subnet_sample)

        subnet_sample = param_dict
        sample = param_dict
        # Write result for csv_path
        if self.csv_path:
            with open(self.csv_path, 'a') as f:
                writer = csv.writer(f)
                date = str(datetime.now())
                result = [
                    subnet_sample,
                    date,
                    individual_results['latency'],
                    individual_results['macs'],
                    individual_results['params'],
                    individual_results['bleu'],
                ]
                writer.writerow(result)

        # PyMoo only minimizes objectives, thus accuracy needs to be negative
        individual_results['bleu'] = -individual_results['bleu']
        # Return results to pymoo
        if len(self.optimization_metrics) == 1:
            return sample, individual_results[self.optimization_metrics[0]]
        elif len(self.optimization_metrics) == 2:
            return (
                sample,
                individual_results[self.optimization_metrics[0]],
                individual_results[self.optimization_metrics[1]],
            )
        elif len(self.optimization_metrics) == 3:
            return (
                sample,
                individual_results[self.optimization_metrics[0]],
                individual_results[self.optimization_metrics[1]],
                individual_results[self.optimization_metrics[2]],
            )
        else:
            log.error('Number of optimization_metrics is out of range. 1-3 supported.')
            return None
