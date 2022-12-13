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
