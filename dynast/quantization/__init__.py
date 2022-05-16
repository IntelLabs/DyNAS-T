import os

from dynast.utils import get_hostname, samples_to_batch_multiply
from dynast.utils.nn import validate_classification
from dynast.utils.ov import load_openvino, save_openvino, save_ov_quantized


def quantize_ov(model, img_size, experiment_name=None, quant_policy: str = 'DefaultQuantization', stat_subset_size: int = None):
    if not experiment_name:
        experiment_name = '{}_dynast_eval'.format(get_hostname())
    if not stat_subset_size:
        stat_subset_size = samples_to_batch_multiply(300, img_size[0])

    folder_name = os.path.expanduser('/store/.torch/{}'.format(experiment_name))
    ov_model_dir = os.path.join(folder_name, 'ov_model')

    save_openvino(
        model,
        img_size,
        ov_model_dir,
        experiment_name
    )

    save_ov_quantized(
        tmp_folder=ov_model_dir,
        model_name=experiment_name,
        quant_policy=quant_policy,
        stat_subset_size=stat_subset_size
    )


def validate(experiment_name=None, test_size=None, batch_size=128):

    if not experiment_name:
        experiment_name = '{}_dynast_eval'.format(get_hostname())

    folder_name = os.path.expanduser('/store/.torch/{}'.format(experiment_name))
    ov_model_dir = os.path.join(folder_name, 'ov_model')

    subnet_ov_int8 = load_openvino(folder=ov_model_dir, name=experiment_name, is_quantized=True)

    loss_ov_quant, top1_ov_quant, top5_ov_quant = validate_classification(
        net=subnet_ov_int8,
        is_openvino=True,
        test_size=test_size,
        batch_size=batch_size,
    )
    return top1_ov_quant, top5_ov_quant
