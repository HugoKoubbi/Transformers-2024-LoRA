# %%
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
# ranks=[]
eigenvalues_all=[]
Ws=[]
for ckpt in all_ckpts:
    model=AutoModelForCausalLM.from_pretrained(ckpt,token=hf_token,device_map="cpu")
    print(f'Checking weights on checkpoint {ckpt.split("-")[-1]}')
    ranks_ckpt=[]
    eigenvalues_ckpt=[]
    for i,block in enumerate(model.model.layers):
        print(f'Block {i}')
        lora_A=block.self_attn.v_proj.lora_A.default.weight.detach().numpy()
        lora_B=block.self_attn.v_proj.lora_B.default.weight.detach().numpy()
        deltaW=lora_B@lora_A
        Ws.append(deltaW)
        # rank=np.linalg.matrix_rank(deltaW)
        # ranks_ckpt.append(rank)
        eigenvalues, eigenvectors = LA.eig(deltaW)
        # x=eigenvalues.real
        # y=eigenvalues.imag
        # plt.figure()
        # plt.scatter(x,y)
        # plt.axis('equal')
        # plt.show()
        eigenvalues_great=eigenvalues[np.argsort(np.abs(eigenvalues))][-2:]
        eigenvalues_ckpt.append(eigenvalues_great)
        print(eigenvalues_great)
        print('---------------------------------')
    # ranks.append(ranks_ckpt)
    eigenvalues_all.append(eigenvalues_ckpt)
    del model
    print('\n---------------------------------------------\n---------------------------------------------\n')
eigenvalues_all=np.array(eigenvalues_all)
# ranks=np.array(ranks)
if not os.path.exists('/npfiles/finetuning'):
    os.makedirs('/npfiles/finetuning')
np.save('/npfiles/finetuning/eigenvalues_all.npy',eigenvalues_all)
np.save('/npfiles/finetuning/Ws.npy',Ws)


# %%

# %%



