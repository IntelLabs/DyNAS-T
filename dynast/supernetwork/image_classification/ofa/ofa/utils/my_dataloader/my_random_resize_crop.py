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


import random
import time

import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image

__all__ = ["MyRandomResizedCrop"]

_pil_interpolation_to_str = {
    Image.NEAREST: "PIL.Image.NEAREST",
    Image.BILINEAR: "PIL.Image.BILINEAR",
    Image.BICUBIC: "PIL.Image.BICUBIC",
    Image.LANCZOS: "PIL.Image.LANCZOS",
    Image.HAMMING: "PIL.Image.HAMMING",
    Image.BOX: "PIL.Image.BOX",
}


class MyRandomResizedCrop(transforms.RandomResizedCrop):
    ACTIVE_SIZE = 224
    IMAGE_SIZE_LIST = [224]
    IMAGE_SIZE_SEG = 4

    CONTINUOUS = False
    SYNC_DISTRIBUTED = True

    EPOCH = 0
    BATCH = 0

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=Image.BILINEAR,
    ):
        if not isinstance(size, int):
            size = size[0]
        super(MyRandomResizedCrop, self).__init__(size, scale, ratio, interpolation)

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(
            img,
            i,
            j,
            h,
            w,
            (MyRandomResizedCrop.ACTIVE_SIZE, MyRandomResizedCrop.ACTIVE_SIZE),
            self.interpolation,
        )

    @staticmethod
    def get_candidate_image_size():
        if MyRandomResizedCrop.CONTINUOUS:
            min_size = min(MyRandomResizedCrop.IMAGE_SIZE_LIST)
            max_size = max(MyRandomResizedCrop.IMAGE_SIZE_LIST)
            candidate_sizes = []
            for i in range(min_size, max_size + 1):
                if i % MyRandomResizedCrop.IMAGE_SIZE_SEG == 0:
                    candidate_sizes.append(i)
        else:
            candidate_sizes = MyRandomResizedCrop.IMAGE_SIZE_LIST

        relative_probs = None
        return candidate_sizes, relative_probs

    @staticmethod
    def sample_image_size(batch_id=None):
        if batch_id is None:
            batch_id = MyRandomResizedCrop.BATCH
        if MyRandomResizedCrop.SYNC_DISTRIBUTED:
            _seed = int("%d%.3d" % (batch_id, MyRandomResizedCrop.EPOCH))
        else:
            _seed = os.getpid() + time.time()
        random.seed(_seed)
        candidate_sizes, relative_probs = MyRandomResizedCrop.get_candidate_image_size()
        MyRandomResizedCrop.ACTIVE_SIZE = random.choices(candidate_sizes, weights=relative_probs)[0]

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + "(size={0}".format(MyRandomResizedCrop.IMAGE_SIZE_LIST)
        if MyRandomResizedCrop.CONTINUOUS:
            format_string += "@continuous"
        format_string += ", scale={0}".format(tuple(round(s, 4) for s in self.scale))
        format_string += ", ratio={0}".format(tuple(round(r, 4) for r in self.ratio))
        format_string += ", interpolation={0})".format(interpolate_str)
        return format_string
