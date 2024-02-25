
from transformers import PreTrainedModel,PreTrainedTokenizerBase,DataCollatorForSeq2Seq
from transformers.file_utils import PaddingStrategy

    
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union


@dataclass
class MeMDataCollator(DataCollatorForSeq2Seq):

    embedding_tokenizer: PreTrainedTokenizerBase = None
    
    def __call__(self, features, return_tensors=None):
        history_ids = (
            [feature['history_input_ids'] for feature in features]
            if 'history_input_ids' in features[0].keys()
            else None
        )
        if history_ids is not None:
            max_history_length = max(len(out) for out in history_ids)
            if self.pad_to_multiple_of is not None:
                max_history_length = (
                        (
                                max_history_length + self.pad_to_multiple_of - 1) //
                        self.pad_to_multiple_of * self.pad_to_multiple_of
                )
            for feature in features:
                remainder = [self.embedding_tokenizer.pad_token_id] * (
                        max_history_length - len(feature['history_input_ids'])
                )
                feature['history_input_ids'] = feature['history_input_ids'] + remainder
                feature['history_attention_mask'] = feature['history_attention_mask'] + [0]*len(remainder)
                feature['history_token_type_ids'] = feature['history_token_type_ids'] + [0]*len(remainder)
        return super().__call__(features, return_tensors)
    
    
# from transformers import LlamaTokenizer
# tokenizer = LlamaTokenizer.from_pretrained('daryl149/llama-2-7b-hf')
# datacol=MeMDataCollator(tokenizer=tokenizer,embedding_tokenizer=tokenizer)
# datacol([{'input_ids':"your are a sentence!",'history':"no history!"}])