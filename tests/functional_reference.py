from dynast.utils.datasets import Dataset
from dynast.utils.reference import TorchVisionReference


def test_reference_torch_accuracy(device, dataset_name, batch_size):
    ref = TorchVisionReference(
        model_name="resnet50",
        dataset=Dataset.get(dataset_name),
    )

    # Note: Accuracy will be relatively high due to small text size.
    _, top1, _ = ref.validate(
        device=device,
        batch_size=batch_size,
        test_size=10,
    )

    expected_top1_range = (88.0, 89.0)

    # There might be slight variations in accuracy, depending on the backend, torch version etc.,
    # so we check if the result is within 1% error.
    assert min(expected_top1_range) <= top1 <= max(expected_top1_range)
