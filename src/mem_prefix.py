# Copyright 2024-present
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

# Based on https://github.com/THUDM/P-tuning-v2/blob/main/model/prefix_encoder.py
# with some refactor

import torch
from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class MemPrefixConfig():
    """
    This is the configuration class to store the configuration of a [`MemPrefixEncoder`].
    """

    num_virtual_tokens: int = field(default=None, metadata={"help": "Number of virtual tokens"})
    token_dim: int = field(
        default=None, metadata={"help": "The hidden embedding dimension of the base transformer model"}
    )
    num_attention_heads: Optional[int] = field(default=None, metadata={"help": "Number of attention heads"})
    num_layers: Optional[int] = field(default=None, metadata={"help": "Number of transformer layers"})
    encoder_hidden_size: int = field(
        default=None,
        metadata={"help": "The hidden size of the encoder"},
    )
    embedding_model_name:  Optional[str] = field(
        default=None, metadata={"help": "The name of the embedding model to use."}
    )
    embedding_dim: int = field(
        default=None,
        metadata={"help": "The hidden embedding dimension of the embedding model"},
    )
    
    


class MemPrefixEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        token_dim = config.token_dim
        num_layers = config.num_layers
        num_virtual_tokens = config.num_virtual_tokens
        encoder_hidden_size = config.encoder_hidden_size
        
        embedding_model_name=config.embedding_model_name
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name)
        embedding_dim=config.embedding_dim
        
        # use a two layer MLP to transform the latent embeddings to tokens
        # self.transform = torch.nn.Sequential(
        #     torch.nn.Linear(embedding_dim, encoder_hidden_size),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(encoder_hidden_size, num_virtual_tokens*num_layers * 2 * token_dim),
        # )
        # first use a linear layer
        self.transform = torch.nn.Linear(embedding_dim, num_virtual_tokens*num_layers * 2 * token_dim)
        

    def mean_pooling(self,token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings
    
    def forward(self,history_input_ids,history_token_type_ids,history_attention_mask):
        input={
            'input_ids':history_input_ids,
            'token_type_ids':history_token_type_ids,
            'attention_mask':history_attention_mask
        }
        outputs = self.embedding_model(**input)
        prefix_embeddings = self.mean_pooling(outputs[0], history_attention_mask)
        past_key_values = self.transform(prefix_embeddings)
        return past_key_values

    @staticmethod
    def tokenize(config,embedding_tokenizer,input):
        #if the length excess the max token length TODO
        result=embedding_tokenizer(input, padding=False, max_length=config.embedding_cutoff_len,truncation=True, return_tensors=None)
        return result
    
# config = MemPrefixConfig(
#     num_virtual_tokens=20,
#     token_dim=768,
#     num_attention_heads=12,
#     num_layers=12,
#     encoder_hidden_size=768,
#     embedding_model_name='facebook/contriever',
#     embedding_dim=768
# )

# mp_encoder=MemPrefixEncoder(config)
# prefix=mp_encoder.tokenize(["I am a sentence.","hello world!"])
# past_key_values=mp_encoder(prefix['input_ids'],prefix['token_type_ids'],prefix['attention_mask'])
# print(past_key_values.shape)
