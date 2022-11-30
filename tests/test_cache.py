# INTEL CONFIDENTIAL
# Copyright 2022 Intel Corporation. All rights reserved.

# This software and the related documents are Intel copyrighted materials, and your use of them is governed by the
# express license under which they were provided to you ("License"). Unless the License provides otherwise, you may
# not use, modify, copy, publish, distribute, disclose or transmit this software or the related documents without
# Intel's prior written permission.

# This software and the related documents are provided as is, with no express or implied warranties, other than those
# that are expressly stated in the License.

# This software is subject to the terms and conditions entered into between the parties.

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
