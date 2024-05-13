
#%%
from unsloth.models import FastLanguageModel
import torch
from dataclasses import dataclass, field

#%%
@dataclass
class ModelConfig:
    name: str = field(default="unsloth/mistral-7b-bnb-4bit")
    max_seq_length: int = field(default=4096)
    dtype: torch.dtype = field(default=None)
    load_in_4bit: bool = field(default=True)

NAME = "unsloth/mistral-7b-bnb-4bit"

four_bit_model = FastLanguageModel.from_pretrained(

    model_name     = "unsloth/mistral-7b-bnb-4bit",
    max_seq_length = 4096,
    dtype          = None,
    load_in_4bit   = True,
    token          = None,
    device_map     = "sequential",
    rope_scaling   = None,
    fix_tokenizer  = True,
    trust_remote_code = False,
    use_gradient_checkpointing = True,
    resize_model_vocab = None

)


