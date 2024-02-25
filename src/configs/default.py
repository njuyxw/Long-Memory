from dataclasses import dataclass

@dataclass
class DefaultConfig:
    # model/data params
    base_model: str = "daryl149/llama-2-7b-hf"
    data_path: str = "../data/ultra_chat/ultra_chat.py"
    output_dir: str = "./outputs/r"
    data_num: int = 10100

    # training hyperparams
    batch_size: int = 128
    micro_batch_size: int = 4
    num_epochs: int = 3
    learning_rate: float = 3e-4
    cutoff_len: int = 512
    embedding_cutoff_len: int = 512
    val_set_size: int = 100

    # llm hyperparams
    train_on_inputs: bool = True
    add_eos_token: bool = True
    group_by_length: bool = False
    
    # mem prefix config
    num_virtual_tokens: int =5
    token_dim: int =None
    num_attention_heads: int =None
    num_layers: int =None
    encoder_hidden_size: int =768
    embedding_model_name: str ="facebook/contriever"
    embedding_dim: int =768
