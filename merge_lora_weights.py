# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

int8 = (os.environ.get('INT8','false') == 'true')

def main(base_model: str,
         peft_model: str,
         output_dir: str):
        
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=int8,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="tmp", 
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model
    )
        
    model = PeftModel.from_pretrained(
        model, 
        peft_model, 
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="tmp",
    )

    model = model.merge_and_unload()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(main)