import pytest

from dynast.search.search_tactic import NASBaseConfig
from dynast.utils.exceptions import InvalidMetricsException, InvalidSupernetException


class TestNASBaseConfig:
    def test_verify_measurement_types_all_valid(self):
        nbc = NASBaseConfig(
            supernet='ofa_mbv3_d234_e346_k357_w1.0',
            optimization_metrics=['latency', 'accuracy_top1'],
            measurements=['latency', 'macs', 'params', 'accuracy_top1'],
        )
        assert list(sorted(nbc.measurements)) == list(sorted(['latency', 'macs', 'params', 'accuracy_top1']))
        assert list(sorted(nbc.optimization_metrics)) == list(sorted(['latency', 'accuracy_top1']))

    def test_verify_measurement_types_invalid_optimization_metric_raises_exception(self):
        with pytest.raises(InvalidMetricsException):
            nbc = NASBaseConfig(
                supernet='ofa_mbv3_d234_e346_k357_w1.0',
                optimization_metrics=['fake_metric', 'accuracy_top1'],
                measurements=['latency', 'macs', 'params', 'accuracy_top1'],
            )

    def test_verify_measurement_types_invalid_measurements_raises_exception(self):
        with pytest.raises(InvalidMetricsException):
            nbc = NASBaseConfig(
                supernet='ofa_mbv3_d234_e346_k357_w1.0',
                optimization_metrics=['latency', 'accuracy_top1'],
                measurements=['fake_metric', 'macs', 'params', 'accuracy_top1'],
            )

    def test_verify_measurement_types_invalid_supernet(self):
        with pytest.raises(InvalidSupernetException):
            nbc = NASBaseConfig(
                supernet='fake_supernet',
                optimization_metrics=['latency', 'accuracy_top1'],
                measurements=['latency', 'macs', 'params', 'accuracy_top1'],
            )
