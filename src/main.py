

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

from pytorch_lightning import seed_everything


from datasets import load_dataset
from configs import DefaultConfig
from prompt import LLamaPromptTemplate
from mem_prefix import MemPrefixEncoder
from model import MemModel
from collector import MeMDataCollator
from utils import _prepare_prompt_learning_config
def train():
    seed_everything(42)
    configs=DefaultConfig()
    # guanaco_dataset = "mlabonne/guanaco-llama2-1k"
    # dataset = load_dataset(guanaco_dataset, split="train")
    dataset=load_dataset(configs.data_path)["train"]

    if configs.data_num!=None:
        dataset = dataset.select(range(configs.data_num))
        
    #quant config
    # compute_dtype = getattr(torch, "float16")
    # quant_config = BitsAndBytesConfig(
    # load_in_4bit=True,
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype=compute_dtype,
    # bnb_4bit_use_double_quant=False,
    # )
    
    
    # model = AutoModelForCausalLM.from_pretrained(
    # configs.base_model,
    # quantization_config=quant_config,
    # device_map = "auto"
    # )
    device_map='auto'
    model = AutoModelForCausalLM.from_pretrained(
        configs.base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model.config.use_cache = True
    model.config.pretraining_tp = 1
    
    configs=_prepare_prompt_learning_config(configs,model.config.to_dict())
    
    embedding_tokenizer = AutoTokenizer.from_pretrained(configs.embedding_model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(configs.base_model, trust_remote_code=True)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt, truncation=True, max_length=configs.cutoff_len,padding=False,return_tensors=None)
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < configs.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        # constract history prompt
        result={}

        history_prompter=LLamaPromptTemplate()
        for i in range(len(data_point['conversations'][:-2])):
            _from=data_point['conversations'][i]['from']
            _value=data_point['conversations'][i]['value']
            assert (_from=='human' and i%2==0) or (_from=='gpt' and i%2==1)
            if _from=='human':
                history_prompter.add_user_message(_value)
            elif _from=='gpt':
                history_prompter.add_model_reply(_value)
        history_prompt=history_prompter.build_prompt()
        # tokenize history 
        # may be wrong because the template is for llama tokenizer but we choose the contriever tokenizer
        history_tokenized=MemPrefixEncoder.tokenize(configs,embedding_tokenizer,history_prompt)
        result['history_input_ids']=history_tokenized['input_ids']
        result['history_token_type_ids']=history_tokenized['token_type_ids']
        result['history_attention_mask']=history_tokenized['attention_mask']
        
        # construct model input
        user_q=data_point['conversations'][-2]['value']
        assit_a=data_point['conversations'][-1]['value']
        prompter=LLamaPromptTemplate()
        user_prompt=prompter.add_user_message(user_q,return_prompt=True)
        prompter.add_model_reply(assit_a)
        full_prompt=prompter.build_prompt()
        tokenized_full_prompt = tokenize(full_prompt)
        tokenized_user_prompt = tokenize(
            user_prompt, add_eos_token=configs.add_eos_token
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        if configs.add_eos_token:
            user_prompt_len -= 1
        result["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  
        result['input_ids']=tokenized_full_prompt['input_ids']
        result['attention_mask']=tokenized_full_prompt['attention_mask']
        return result
    
    #construct dataset
    if configs.val_set_size > 0:
        train_val = dataset.train_test_split(test_size=configs.val_set_size, shuffle=True, seed=42)
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt,remove_columns='conversations')
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt,remove_columns='conversations')
    else:
        train_data = dataset.shuffle().map(generate_and_tokenize_prompt,remove_columns='conversations')
        val_data = None
    
    mem_model=MemModel(model,configs)
    mem_model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    gradient_accumulation_steps = configs.batch_size // configs.micro_batch_size
    training_params = TrainingArguments(
        output_dir="./results",
        num_train_epochs=configs.num_epochs,
        per_device_train_batch_size=configs.micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="adamw_torch",
        save_steps=25,
        logging_steps=2,
        learning_rate=configs.learning_rate,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.1,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
        )
    trainer = Trainer(
        model=mem_model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_params,
        data_collator=MeMDataCollator(
            tokenizer=tokenizer, pad_to_multiple_of=8, padding=True,embedding_tokenizer=embedding_tokenizer
        ),
    )
    trainer.train()

if __name__ == "__main__":
    train()
    
    