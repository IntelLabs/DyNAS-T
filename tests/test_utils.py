# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from unittest import mock
from unittest.mock import mock_open, patch

import pytest
import requests

from dynast.utils import get_cores, get_remote_file

valid_remote_url = 'http://someurl.com/test.txt'
valid_remote_url_file_not_exists = 'http://example.com/supernets/not_exists.txt'
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


@mock.patch("subprocess.Popen")
def test_get_cores(mock_subproc_popen):
    with open('tests/outputs/subprocess_lscpu_icx8380.json', 'r') as f:
        content = f.read()
    process_mock = mock.Mock()
    attrs = {"communicate.return_value": (content, None)}
    process_mock.configure_mock(**attrs)
    mock_subproc_popen.return_value = process_mock

    # Get all logical cores
    cores = [int(c) for c in get_cores().split(',')]
    assert len(cores) == 160
    assert list(range(160)) == sorted(cores)

    # Get all logical cores from socket #1 only
    cores = [int(c) for c in get_cores(sockets=[0]).split(',')]
    assert len(cores) == 80
    assert (list(range(0, 40)) + list(range(80, 120))) == sorted(cores)

    # Get physical cores only
    cores = [int(c) for c in get_cores(use_ht=False).split(',')]
    assert len(cores) == 80
    assert list(range(0, 80)) == sorted(cores)

    # Get physical cores from socket #1 only
    cores = [int(c) for c in get_cores(use_ht=False, sockets=[1]).split(',')]
    assert len(cores) == 40

    # Get first 20 physical cores from socket #1 only
    cores = [int(c) for c in get_cores(use_ht=False, sockets=[1], num_cores=20).split(',')]
    assert len(cores) == 20
