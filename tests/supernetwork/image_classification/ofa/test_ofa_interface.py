import pytest

from dynast.supernetwork.image_classification.ofa.ofa_interface import OFARunner


class TestOFARunner:
    def test_init_data_path_not_exist(self):
        with pytest.raises(FileNotFoundError):
            OFARunner(
                supernet='ofa_resnet50',
                dataset_path='/not-existing-path-to-a-dataset',
            )

    def test_init_data_path_none(self):
        OFARunner(
            supernet='ofa_resnet50',
            dataset_path=None,
        )
