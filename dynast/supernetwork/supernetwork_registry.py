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


from dynast.supernetwork import SupernetBaseRegisteredClass, SupernetRegistryHolder
from dynast.supernetwork.image_classification.ofa import OFAResNet50Supernet
from dynast.supernetwork.image_classification.ofa.ofa_encoding import OFAMobileNetV3Encoding, OFAResNet50Encoding
from dynast.supernetwork.image_classification.ofa.ofa_interface import (
    EvaluationInterfaceOFAMobileNetV3,
    EvaluationInterfaceOFAResNet50,
)
from dynast.supernetwork.machine_translation.transformer_encoding import TransformerLTEncoding
from dynast.supernetwork.machine_translation.transformer_interface import EvaluationInterfaceTransformerLT
from dynast.supernetwork.text_classification.bert_encoding import BertSST2Encoding
from dynast.supernetwork.text_classification.bert_interface import EvaluationInterfaceBertSST2




def main():
    print([s for s in SupernetRegistryHolder.get_registry()])


if __name__ == '__main__':
    main()
