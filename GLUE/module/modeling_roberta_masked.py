# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Masked RoBERTa model. """

from transformers.models.roberta.modeling_roberta import RobertaEmbeddings
from transformers.models.roberta.configuration_roberta import RobertaConfig
from .modeling_bert_masked import MaskedBertModel


class MaskedRobertaModel(MaskedBertModel):

    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(MaskedRobertaModel, self).__init__(config)
        self.embeddings = RobertaEmbeddings(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
