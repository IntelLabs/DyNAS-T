from dynast.supernetwork.supernetwork_registry import SUPERNET_METRICS, SUPERNET_PARAMETERS


class InvalidSupernetException(Exception):
    def __init__(self, supernet):
        self.message = f'Invalid super-network ({supernet}) specified. Choose from the following: {list(SUPERNET_PARAMETERS.keys())}'
        super().__init__(self.message)


class InvalidMetricsException(Exception):
    def __init__(self, supernet, metric):
        valid_metrics = SUPERNET_METRICS[supernet]
        self.message = (
            f'Invalid metric specified: {metric}. Super-network f{supernet} supports following metrics: {valid_metrics}'
        )
        super().__init__(self.message)
