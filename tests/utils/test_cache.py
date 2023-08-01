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

from dynast.utils.cache import Cache


def test_cache_add_new(tmp_path):
    cache = Cache("ut", tmp_path)
    payload = {"a": 1, "b": 2}
    cache.update(key="test1", payload=payload)
    assert cache.get(key="test1") == payload


def test_cache_update_existing(tmp_path):
    cache = Cache("ut", tmp_path)
    payload = {"a": 1, "b": 2}
    cache.update(key="test1", payload=payload)

    updated_payload = {"a": 1, "b": "b", "c": 3}
    cache.update(key="test1", payload=updated_payload)

    assert cache.get(key="test1") == updated_payload


def test_cache_get_missing_key_none(tmp_path):
    cache = Cache("ut", tmp_path)
    payload = {"a": 1, "b": 2}
    cache.update(key="test1", payload=payload)

    assert cache.get(key="test2") == None


def test_key(tmp_path):
    cache = Cache("ut", tmp_path)
    assert cache.key([0, 0, 0, 0, 1, 1]) == "0_0_0_0_1_1"
