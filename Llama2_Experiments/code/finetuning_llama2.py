#%%
from datasets import load_dataset,ClassLabel, Value
from transformers import AutoTokenizer,AutoModelForCausalLM,TrainingArguments,BitsAndBytesConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import os
from peft import LoraConfig
import torch
import bitsandbytes as bnb
import wandb
access_token=os.environ["HF_AUTH_TOKEN"]
wandb.login(key=os.environ["WANDB_API_KEY"])
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf",token=access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model=AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",token=access_token,quantization_config=quant_config,device_map='auto')

response_template_with_context = "\n### Assistant:"  # We added context here: "\n". This is enough for this tokenizer
response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
imdb = load_dataset("imdb",split="train")
imdb=imdb.rename_columns({"text":"prompt","label":"completion"})
# %%
id2label = {'0': "NEGATIVE", '1': "POSITIVE"}
imdb_transformed=imdb.cast_column("completion",Value("string"))
def process_prompt(example):
    example['prompt']= f'''
    Say if this critic is positive or negative, reply with "POSITIVE" or "NEGATIVE" only.
    ###CRITIC: {example['prompt'].replace("<br />","")}
    '''
    example['completion']=id2label[example['completion']]
    return example

def formatting_func(example):
    outputs=[]
    for i in range(len(example['prompt'])):
        res= f'''
        {example['prompt'][i]}
        \n### Assistant: {example['completion'][i]} 
        '''
        outputs.append(res)
    return outputs
    
peft_config = LoraConfig(
    r=2,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['v_proj']
)
imdb_transformed=imdb_transformed.map(process_prompt)
imdb_transformed=imdb_transformed.shuffle(seed=42)
output_dir="/ckpt/rerun"
args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=3,
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="wandb"
)

trainer = SFTTrainer(
    model,
    args=args,
    train_dataset=imdb_transformed,
    data_collator=data_collator,
    formatting_func=formatting_func,
    peft_config=peft_config,
)

# %%
resume_from_checkpoint=len(os.listdir(output_dir))>0
trainer.train(resume_from_checkpoint=resume_from_checkpoint)
# %%
