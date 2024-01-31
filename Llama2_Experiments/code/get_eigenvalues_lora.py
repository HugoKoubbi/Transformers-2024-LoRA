from transformers import AutoTokenizer,AutoModelForCausalLM
import os as os
import numpy as np
import matplotlib.pyplot as plt
import boto3
import transformers
import peft

# %%
s3 = boto3.resource('s3')
# Print out bucket names
hf_token=os.environ["HF_AUTH_TOKEN"]
bucket=s3.Bucket('llama-imdb-chkpts')

for object in bucket.objects.all():
    if object.size!=0:
        base,key=os.path.split(object.key)
        if "rerun" in base:
            continue
        if not os.path.exists(f"../chkpt/{base}"):
            os.makedirs(f"../chkpt/{base}")
        bucket.download_file(object.key, f"../chkpt/{object.key}")

# %%
from numpy import linalg as LA
num_ckpts=len(os.listdir("../chkpt/"))
all_ckpts=[f"../chkpt/checkpoint-{25*i}" for i in range(1,num_ckpts+1)]
last_ckpt=all_ckpts[-1]
lora_model=AutoModelForCausalLM.from_pretrained(last_ckpt,token=hf_token,device_map="cpu")
eigenvalues_tilde_all=[]
eigenvalues_all=[]
for i,block in enumerate(lora_model.model.layers):
    print(f'Block {i}')
    lora_A=block.self_attn.v_proj.lora_A.default.weight.detach().numpy()
    lora_B=block.self_attn.v_proj.lora_B.default.weight.detach().numpy()
    v=block.self_attn.v_proj.weight.detach().numpy()
    deltaW=lora_B@lora_A
    v_tilde=v+deltaW
    eigenvalues_tilde,_ = LA.eig(v_tilde)
    eigenvalues,_ = LA.eig(v)
    eigenvalues_tilde_all.append(eigenvalues_tilde)
    eigenvalues_all.append(eigenvalues)
    print('---------------------------------')
np.save("/npfiles/eigenvalues_tilde_all.npy",eigenvalues_tilde_all)
np.save("/npfiles/eigenvalues_all.npy",eigenvalues_all)