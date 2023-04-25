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

from typing import Dict


class SupernetRegistryHolder(type):

    _REGISTRY: Dict[str, "SupernetRegistryHolder"] = {}

    def __new__(cls, name, bases, attrs):
        new_cls = type.__new__(cls, name, bases, attrs)
        print(new_cls.__name__)
        cls._REGISTRY[new_cls._name] = new_cls()
        return new_cls

    @classmethod
    def get_registry(cls):
        return {
            k: v for k, v in cls._REGISTRY.items() if k not in ['SupernetBaseRegisteredClass', 'dynast.supernetwork']
        }


class SupernetBaseRegisteredClass(metaclass=SupernetRegistryHolder):
    _name = __name__
    _encoding = None
    _parameters = None
    _evaluation_interface = None
    _linas_innerloop_evals = None
    _supernet_type = None
    _supernet_metrics = None

    @property
    def encoding(self):
        return self._encoding

    @property
    def parameters(self):
        return self._parameters

    @property
    def evaluation_interface(self):
        return self._evaluation_interface

    @property
    def linas_innerloop_evals(self):
        return self._linas_innerloop_evals

    @property
    def supernet_type(self):
        return self._supernet_type

    @property
    def supernet_metrics(self):
        return self._supernet_metrics

    def __str__(self) -> str:
        return self._name
