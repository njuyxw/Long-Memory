# Copyright 2024-present
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

from typing import Any,List, Optional, Union
import os
from utils import _get_batch_size,_prepare_prompt_learning_config
import warnings

import torch
from transformers import PreTrainedModel
from mem_prefix import MemPrefixConfig,MemPrefixEncoder

class MemModel(torch.nn.Module):
    def __init__(self, model: PreTrainedModel, configs) -> None:
        super().__init__()
        self.modules_to_save = None
        self.configs = configs
        self.base_model = model
        self.prompt_encoder=self._get_prompt_encoder(configs)
        self.device=model.device
        self._set_up_prompt_encoder()
        if getattr(model, "is_gradient_checkpointing", True):
            model = self._prepare_model_for_gradient_checkpointing(model)
            
    def _prepare_model_for_gradient_checkpointing(self, model: PreTrainedModel):
        r"""
        Prepares the model for gradient checkpointing if necessary
        """
        if not (
            getattr(model, "is_loaded_in_8bit", False)
            or getattr(model, "is_loaded_in_4bit", False)
            or getattr(model, "is_quantized", False)
        ):
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            elif hasattr(model, "get_input_embeddings"):

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        return model
    
    def _get_prompt_encoder(self,configs):
        mem_config = MemPrefixConfig(
            num_virtual_tokens=configs.num_virtual_tokens,
            token_dim=configs.token_dim,
            num_attention_heads=configs.num_attention_heads,
            num_layers=configs.num_layers,
            encoder_hidden_size=configs.encoder_hidden_size,
            embedding_model_name=configs.embedding_model_name,
            embedding_dim=configs.embedding_dim,
        )
        return MemPrefixEncoder(mem_config)
        
    def _set_up_prompt_encoder(self):
        transformer_backbone = None
        for name, module in self.base_model.named_children():
            for param in module.parameters():
                param.requires_grad = False
            if isinstance(module, PreTrainedModel):
                # Make sure to freeze Tranformers model
                if transformer_backbone is None:
                    transformer_backbone = module
                    self.transformer_backbone_name = name
        if transformer_backbone is None:
            transformer_backbone = self.base_model
            
        for named_param, value in list(transformer_backbone.named_parameters()):
            # for ZeRO-3, the tensor is sharded across accelerators and deepspeed modifies it to a tensor with shape [0]
            # the actual unsharded shape is stored in "ds_shape" attribute
            # special handling is needed in case the model is initialized in deepspeed.zero.Init() context or HfDeepSpeedConfig
            # has been called before
            # For reference refer to issue: https://github.com/huggingface/peft/issues/996
            deepspeed_distributed_tensor_shape = getattr(value, "ds_shape", None)

            if value.shape[0] == self.base_model.config.vocab_size or (
                deepspeed_distributed_tensor_shape is not None
                and deepspeed_distributed_tensor_shape[0] == self.base_model.config.vocab_size
            ):
                self.word_embeddings = transformer_backbone.get_submodule(named_param.replace(".weight", ""))
                break
        self.prompt_encoder = self.prompt_encoder.to(self.device)
        
    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        r"""
        Returns the number of trainable parameters and the number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    def print_trainable_parameters(self) -> None:
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params, all_param = self.get_nb_trainable_parameters()

        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )
        
    @property
    def base_model_torch_dtype(self):
        return getattr(self.base_model, "dtype", None)
    
    def get_prompt(self,history_input_ids, history_token_type_ids,history_attention_mask,batch_size: int) -> torch.Tensor:

        peft_config = self.configs
        prompt_encoder = self.prompt_encoder

        past_key_values = prompt_encoder(history_input_ids,history_token_type_ids,history_attention_mask)
        if self.base_model_torch_dtype is not None:
            past_key_values = past_key_values.to(self.base_model_torch_dtype)

        
        past_key_values = past_key_values.view(
            batch_size,
            peft_config.num_virtual_tokens,
            peft_config.num_layers * 2,
            peft_config.num_attention_heads,
            peft_config.token_dim // peft_config.num_attention_heads,
        )
       
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        
        return past_key_values

    def forward(
            self,
            history_input_ids=None,
            history_token_type_ids=None,
            history_attention_mask=None,
            input_ids=None,
            attention_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
        ):
            peft_config = self.configs

            batch_size = _get_batch_size(input_ids, inputs_embeds)
            if attention_mask is not None:
                # concat prompt attention mask
                prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
                attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

            if kwargs.get("position_ids", None) is not None:
                warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
                kwargs["position_ids"] = None
            if kwargs.get("token_type_ids", None) is not None:
                warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
                kwargs["token_type_ids"] = None
            kwargs.update(
                {
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "output_attentions": output_attentions,
                    "output_hidden_states": output_hidden_states,
                    "return_dict": return_dict,
                }
            )

            past_key_values = self.get_prompt(history_input_ids,history_token_type_ids,history_attention_mask,batch_size)
            # print(past_key_values[0].shape,past_key_values[1].shape)
            # print(attention_mask.shape)
            return self.base_model(
                input_ids=input_ids, inputs_embeds=inputs_embeds, past_key_values=past_key_values, **kwargs
            )
        
    def save_mem(
        self,
        save_directory: str,
        is_main_process: bool = True,
        **kwargs: Any,
    ) -> None:
        pass
    
    
    def generate(self, *args, **kwargs):
        raise NotImplementedError
        outputs = self.base_model.generate(*args, **kwargs)
    
    
      