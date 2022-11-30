# INTEL CONFIDENTIAL
# Copyright 2022 Intel Corporation. All rights reserved.

# This software and the related documents are Intel copyrighted materials, and your use of them is governed by the
# express license under which they were provided to you ("License"). Unless the License provides otherwise, you may
# not use, modify, copy, publish, distribute, disclose or transmit this software or the related documents without
# Intel's prior written permission.

# This software and the related documents are provided as is, with no express or implied warranties, other than those
# that are expressly stated in the License.

# This software is subject to the terms and conditions entered into between the parties.


class OVQuantizationPolicy(object):
    @staticmethod
    def _get_base_policy(
        model_name: str,
        fp32_path_xml: str,
        fp32_path_bin: str,
        annotation_file: str = '/store/nosnap/datasets/pot-acc-aware-quant/datasets/ImageNet/original/ILSVRC2012_val.txt',
        data_source: str = '/store/nosnap/datasets/pot-acc-aware-quant/datasets/ImageNet/original/',
        input_size: int = 224,
    ) -> dict:
        return {
            'model': {'model_name': model_name, 'model': fp32_path_xml, 'weights': fp32_path_bin},
            'engine': {
                'launchers': [{'framework': 'dlsdk', 'adapter': 'classification'}],
                'datasets': [
                    {
                        'name': 'imagenet_1000_classes',
                        'reader': 'pillow_imread',
                        'annotation_conversion': {
                            'converter': 'imagenet',
                            'annotation_file': annotation_file,
                        },
                        'data_source': data_source,
                        'preprocessing': [
                            {
                                'type': 'resize',
                                'size': 256,
                                'aspect_ratio_scale': 'greater',
                                'use_pillow': True,
                                'interpolation': 'BILINEAR',
                            },
                            {'type': 'crop', 'size': input_size, 'use_pillow': True},
                            {
                                'type': 'normalization',
                                'mean': [123.675, 116.28, 103.53],  # NOTE Valid only for ImageNet
                                'std': [58.624, 57.12, 57.375],  # NOTE Valid only for ImageNet
                            },
                        ],
                        'metrics': [
                            {'name': 'accuracy@top1', 'type': 'accuracy', 'top_k': 1},
                            {'name': 'accuracy@top5', 'type': 'accuracy', 'top_k': 5},
                        ],
                    }
                ],
            },
            'compression': {
                'target_device': 'CPU',
            },
        }

    @staticmethod
    def get_policy(
        model_name: str,
        fp32_path_xml: str,
        fp32_path_bin: str,
        quant_policy: str = 'DefaultQuantization',
        stat_subset_size: int = 3 * 128,
    ) -> dict:
        assert quant_policy in ['DefaultQuantization', 'AccuracyAwareQuantization']

        policy_to_use = OVQuantizationPolicy._get_base_policy(model_name, fp32_path_xml, fp32_path_bin)
        if quant_policy == 'DefaultQuantization':
            policy_to_use['compression']['algorithms'] = [
                {
                    'name': 'DefaultQuantization',
                    'params': {
                        'preset': 'performance',
                        'stat_subset_size': stat_subset_size,
                    },
                }
            ]
        elif quant_policy == 'AccuracyAwareQuantization':
            policy_to_use['compression']['algorithms'] = [
                {
                    'name': 'AccuracyAwareQuantization',
                    'params': {
                        'preset': 'performance',
                        'stat_subset_size': stat_subset_size,
                        'max_iter_num': 30,
                        'maximal_drop': 0.01,
                        'convert_to_mixed_preset': True,
                    },
                }
            ]
        else:
            raise NotImplementedError('Quantization policy "{}" is not available.'.format(quant_policy))

        return policy_to_use
