import onnxruntime
import torch
import torch.nn as nn


def load_onnx_model(model_path):
    ort_session = onnxruntime.InferenceSession(model_path)
    return ort_session


def save_onnx_model(network, model_path, input_shape):
    device = next(network.parameters()).device
    dummy_data = torch.randn(input_shape, device=device)

    if isinstance(network, nn.DataParallel):
        network = network.module

    # Export the model
    torch.onnx.export(network,  # model being run
                      dummy_data,  # model input (or a tuple for multiple inputs)
                      model_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})
