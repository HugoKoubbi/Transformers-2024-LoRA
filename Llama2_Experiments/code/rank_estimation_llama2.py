#%%
from datasets import load_dataset
from transformers import AutoTokenizer

access_token=os.environ["HF_AUTH_TOKEN"]

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf",token=access_token)
# imdb = load_dataset("imdb")

# def preprocess_function(examples):
#     return tokenizer(examples["text"], truncation=True)

# tokenized_imdb = imdb.map(preprocess_function, batched=True)
# id2label = {0: "NEGATIVE", 1: "POSITIVE"}
# label2id = {"NEGATIVE": 0, "POSITIVE": 1}
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",token=access_token,device_map='cpu')
# %%
from numpy import linalg as LA
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
svds=[]
eigenvalues_all=[]
ranks=[]
for i,block in tqdm(enumerate(model.model.layers)):
    q=block.self_attn.q_proj.weight.detach().numpy()
    k=block.self_attn.k_proj.weight.detach().numpy()
    v=block.self_attn.v_proj.weight.detach().numpy()
    qtk=np.matmul(q,k.T)
    rank=np.linalg.matrix_rank(qtk)
    ranks.append(rank)
    s=LA.svd(v,compute_uv=False)
    svds.append(s)
    eigenvalues, eigenvectors = LA.eig(v)
    eigenvalues_all.append(eigenvalues)
 
# %%
import os

for i in range(len(svds)):
    svd=svds[i]
    eigenvalues=eigenvalues_all[i]
    rank=ranks[i]
    #Saving data for each block in a separate file
    np.save("/npfiles/svd_block"+str(i)+".npy",svd)
    np.save("/npfiles/eigenvalues_block"+str(i)+".npy",eigenvalues)
    with open("/npfiles/rank_block"+str(i)+".txt","w") as f:
        f.write(str(rank))

# %%
