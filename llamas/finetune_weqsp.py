import argparse
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer
from datasets import load_dataset
import os
import torch
from peft import LoraConfig, get_peft_model

import wandb
import huggingface_hub

### Authorization
wandb.login()
os.environ["WANDB_PROJECT"] = "finetune-webqsp-llama"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint" 

access_token = "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"
huggingface_hub.login(token=access_token)
############  

parser = argparse.ArgumentParser()

# Dataset args
parser.add_argument('--base_model', default='meta-llama/Meta-Llama-3-8B-Instruct')
parser.add_argument('--data_dir', default='/home/bo/Dropbox/Projects/kg-llm-uq/llamas/data')
parser.add_argument('--dataset_name', default='webqsp')
parser.add_argument('--output_path', default='/home/bo/Dropbox/Projects/kg-llm-uq/llamas/models')

# Lora args
parser.add_argument('--lora_r', default=8)
parser.add_argument('--lora_alpha', default=16)
parser.add_argument('--lora_dropout', default=0.05)
parser.add_argument('--val_size', default=100)
parser.add_argument('--micro_batch_size', default=2)
parser.add_argument('--epochs', default=2)
parser.add_argument('--lr', default=3e-4)
parser.add_argument('--cutoff_len', default=512)
parser.add_argument('--batch_size', default=32)

parser.add_argument('--seed', default=42)

args = parser.parse_args()

TARGET_MODULES = ['q_proj', 'v_proj']
GRADIENT_ACCUMULATION_STEPS = args.batch_size // args.micro_batch_size

def generate_and_tokenize_prompt(dp, tokenizer):
    # This function masks out the labels for the input
    # Follow practice of alpaca https://github.com/tatsu-lab/stanford_alpaca
    user_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 
    
### Instruction:
{dp['Instruction']} 
    
### Input:
{dp['Input']}

### Response:
"""

    len_user_prompt_tokens = (
        len(
            tokenizer(
                user_prompt,
                truncation=True,
                max_length=args.cutoff_len + 1,
            )["input_ids"]
        )
        - 1
    )  # no eos token
    full_tokens = tokenizer(
        user_prompt + dp["Output"],
        truncation=True, # Let's not truncate
        max_length=args.cutoff_len + 1,
        padding="max_length",
    )["input_ids"][:-1]

    return {
        "input_ids": full_tokens,
        "labels": [-100] * len_user_prompt_tokens
        + full_tokens[len_user_prompt_tokens:],
        "attention_mask": [1] * (len(full_tokens)),
    }

def load_model_and_tokenizer(path_to_model):
    model = LlamaForCausalLM.from_pretrained(
        path_to_model,
        token=access_token,
        torch_dtype=torch.float16).bfloat16().half().cuda()  # Optimized for RTX 4090
    tokenizer = AutoTokenizer.from_pretrained(
        path_to_model 
    )

    return model, tokenizer

def finetune():
    model, tokenizer = load_model_and_tokenizer(args.base_model)
    tokenizer.pad_token_id = 0
    data = load_dataset("json", data_files=f'{args.data_dir}/{args.dataset_name}_finetune_ranking.json', num_proc=3)
    config = LoraConfig(
        r = args.lora_r,
        lora_alpha = args.lora_alpha,
        target_modules = TARGET_MODULES,
        lora_dropout = args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config).cuda()
    
    if args.val_size > 0:
        data = data['train'].train_test_split(test_size=args.val_size, shuffle=True, seed=args.seed) 
        val_data = data['test'].shuffle().map(lambda x: generate_and_tokenize_prompt(x, tokenizer))
    else:
        val_data = None

    train_data = data['train'].shuffle().map(lambda x: generate_and_tokenize_prompt(x, tokenizer))

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=100,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            fp16=True,
            logging_steps=5,
            evaluation_strategy="steps" if args.val_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if args.val_size > 0 else None,
            save_steps=200,
            output_dir=args.output_path,
            save_total_limit=3,
            load_best_model_at_end=True if args.val_size > 0 else False,
            optim="adamw_torch",
            report_to='wandb' 
        ),
        data_collator= DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False
    trainer.train()

    model.save_pretrained(args.output_path)
    
if __name__ == '__main__':
    finetune()