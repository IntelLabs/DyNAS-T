import pytest

from dynast.search_module.search import SearchAlgoManager


def test_search_manager_unsupported_algorithm_raises_NotImplementedError():
    with pytest.raises(NotImplementedError):
        SearchAlgoManager(
            algorithm='super-futuristic-algo',
        )


def test_run_search_unsupported_algorithm_raises_NotImplementedError():
    search_manager = SearchAlgoManager()
    search_manager.algorithm = 'super-futuristic-algo'
    with pytest.raises(NotImplementedError):
        search_manager.run_search(problem=None)
