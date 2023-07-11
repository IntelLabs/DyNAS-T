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


from dynast.utils.nn import AverageMeter


def test_average_meter():
    am = AverageMeter()

    val = 1
    am.update(val=val)
    assert am.val == val
    assert am.count == 1
    assert am.avg == 1

    val = 2
    am.update(val=val, n=2)
    assert am.val == val
    assert am.count == 3
    assert am.avg == 5 / 3
