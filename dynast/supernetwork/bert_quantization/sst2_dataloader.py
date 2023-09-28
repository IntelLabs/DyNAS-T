# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import warnings

import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import Trainer
from transformers.data.processors.glue import glue_processors
from transformers.models.bert.tokenization_bert import BertTokenizer

warnings.filterwarnings("ignore")


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of dictionaries."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for ex_index, example in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[: (max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        features.append(
            {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids, "label_id": label_id}
        )

    return features


def create_tensor_dataset(features):
    batch_input_ids = []
    batch_input_mask = []
    batch_segment_ids = []
    batch_label_ids = []

    for feature in features:
        batch_input_ids.append(feature["input_ids"])
        batch_input_mask.append(feature["input_mask"])
        batch_segment_ids.append(feature["segment_ids"])
        batch_label_ids.append(feature["label_id"])

    return TensorDataset(
        torch.tensor(batch_input_ids, dtype=torch.long),
        torch.tensor(batch_input_mask, dtype=torch.long),
        torch.tensor(batch_segment_ids, dtype=torch.long),
        torch.tensor(batch_label_ids, dtype=torch.long),
    )


def prepare_data_loader(dataset_path, max_seq_length=128, eval_batch_size=16):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    processor = glue_processors["sst-2"]()
    num_labels = len(processor.get_labels())

    eval_examples = processor.get_dev_examples(dataset_path)

    eval_features = convert_examples_to_features(
        eval_examples,
        processor.get_labels(),
        max_seq_length,
        tokenizer,
    )

    eval_data = create_tensor_dataset(eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data,
        sampler=eval_sampler,
        batch_size=eval_batch_size,
    )
    return eval_dataloader


def prepare_calib_loader(dataset_path, model, max_seq_length=128, eval_batch_size=16):
    device = 'cpu'
    raw_datasets = load_dataset("glue", "sst2")
    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)
    label_to_id = None  # {v: i for i, v in enumerate(label_list)}
    padding = "max_length"
    max_seq_length = 128
    sentence1_key = "sentence"
    sentence2_key = None
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    raw_datasets = raw_datasets.map(preprocess_function, batched=True, load_from_cache_file=True)
    eval_dataset = raw_datasets["validation"]
    trainer = Trainer(model=model, train_dataset=None, eval_dataset=eval_dataset, tokenizer=tokenizer)
    calib_dataloader = trainer.get_eval_dataloader()

    return calib_dataloader
