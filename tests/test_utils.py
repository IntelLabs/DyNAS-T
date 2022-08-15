import os
from unittest import mock
from unittest.mock import mock_open, patch

import pytest
import requests

from dynast.utils import get_remote_file

valid_remote_url = 'http://someurl.com/test.txt'
valid_remote_url_file_not_exists = (
    'http://dynas.aipg-rancher-amr.intel.com/supernets/not_exists.txt'
)
existing_model_dir = '/tmp/'
nonexisting_model_dir = '/sup23/123s_nets2'


def mocked_requests_get(*args, **kwargs):
    class MockResponse:
        def __init__(self, content, status_code):
            self.content = content
            self.status_code = status_code

        def get(self):
            return self.content

        def raise_for_status(self):
            print(self.status_code, 'self.status_code ')
            if self.status_code == 404:
                raise requests.exceptions.HTTPError

    if args[0] == valid_remote_url:
        return MockResponse("Hello world!", 200)

    return MockResponse(None, 404)


def test_get_remote_file_model_dir_doesnt_exist_error():
    with pytest.raises(NotADirectoryError):
        get_remote_file(
            remote_url=valid_remote_url,
            model_dir=nonexisting_model_dir,
        )


@patch("builtins.open", new_callable=mock_open, read_data="data")
@mock.patch('requests.get', side_effect=mocked_requests_get)
def test_get_remote_file(mock_get, mock_file):
    get_remote_file(
        remote_url=valid_remote_url,
        model_dir=existing_model_dir,
        overwrite=False,
    )
    mock_file.assert_called_with(os.path.join(existing_model_dir, 'test.txt'), "wb")


@mock.patch('requests.get', side_effect=mocked_requests_get)
def test_get_remote_file_remote_doesnt_exist(mock_get):
    with pytest.raises(requests.exceptions.HTTPError):
        get_remote_file(
            remote_url=valid_remote_url_file_not_exists,
            model_dir=existing_model_dir,
            overwrite=False,
        )
