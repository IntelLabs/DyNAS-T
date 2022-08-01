from typing import BinaryIO, Tuple, Union

import onnxruntime
import torch


def is_onnx(network: object) -> bool:
    """Return whether an object is an instance of onnxruntime.InferenceSession."""
    return isinstance(network, onnxruntime.InferenceSession)


def load_onnx_model(model_path: str) -> onnxruntime.InferenceSession:
    """Create `onnxruntime.InferenceSession` using model stored under `model_path`."""
    ort_session = onnxruntime.InferenceSession(model_path)
    return ort_session


def save_onnx_model(
    network: Union[torch.nn.Module, torch.nn.DataParallel],
    model_path: Union[str, BinaryIO],
    input_shape: Tuple[int, int, int, int],
) -> None:
    """Converts Torch model to ONNX format and saves it to file.

    Arguments:
    ----------
    * network: Torch model
    * model_path: String representing path to a file or a file-like object where output will be saved.
    * input_shape: A tuple(batch_size, channels, resolution, resolution)

    Returns:
    --------
    * None
    """

    device = next(network.parameters()).device
    dummy_data = torch.randn(input_shape, device=device)

    if isinstance(network, torch.nn.DataParallel):
        network = network.module

    # Export the model
    torch.onnx.export(
        network,  # model being run
        dummy_data,  # model input (or a tuple for multiple inputs)
        model_path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable lenght axes
            "output": {0: "batch_size"},
        },
    )
