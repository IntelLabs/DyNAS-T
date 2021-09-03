import pytest

from dynast.common import get_info


def test_get_info():
    assert get_info() == 'dynas-\U0001F375 : '
    assert get_info('cli') == 'dynas-\U0001F375 : cli'
