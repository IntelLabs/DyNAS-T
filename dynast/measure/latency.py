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


def auto_steps(
    batch_size: int,
    is_warmup: bool = False,
    warmup_scale: float = 5.0,
    min_steps: int = 25,
    min_samples: int = 500,
) -> int:
    """Simple scaling of number of steps w.r.t batch_size

    Example:
    1. `auto_steps(1, True),  auto_steps(1, False)` -> 100, 500
    2. `auto_steps(8, True),  auto_steps(8, False)` -> 12, 62
    3. `auto_steps(16, True), auto_steps(8, False)` -> 6, 31
    4. `auto_steps(32, True), auto_steps(8, False)` -> 5, 25

    Args:
    - batch_size
    - is_warmup: if set to True, will scale down the number of steps by `warmup_scale`.
    - warmup_scale: scale by which number of steps should be decreased if `is_warmup` is True.
    - min_steps: minimum number of steps to return
    - min_samples: returned steps multiplied by `batch_size` should be at least this much.
    Returns:
        number of steps
    """

    if not is_warmup:
        warmup_scale = 1.0

    return int(max(batch_size * min_steps, min_samples) // batch_size // warmup_scale)
