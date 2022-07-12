from dynast.utils.datasets import Dataset
from dynast.utils.reference import TorchVisionReference

MODEL_NAME = "resnet50"
DATASET_NAME = "imagenet"
DEVICE = "cpu"
BATCH_SIZE = 128


def test_reference_torch_accuracy():
    ref = TorchVisionReference(
        model_name=MODEL_NAME,
        dataset=Dataset.get(DATASET_NAME),
    )

    # Note: Accuracy will be relatively high due to small text size.
    _, top1, _ = ref.validate(
        device=DEVICE,
        batch_size=BATCH_SIZE,
        test_size=10,
    )

    assert 89.0625 == top1
