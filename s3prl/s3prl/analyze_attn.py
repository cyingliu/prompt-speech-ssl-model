import torch
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    attn_dir = "attn_weight/hubert_asr_pre10_short"
    attn_pths = [os.path.join(attn_dir, f"layer{i}.pt") for i in range(1)]
    prompt_len = 10

    for i in range(1):
        print(i)
        attn = torch.load(attn_pths[i]) # (1, 168, 168)
        attn = attn.squeeze(0).numpy() # (168, 168) (prompt_len + seq_len, prompt_len + seq_len)
        lambdas = []
        for j in range(attn.shape[0]):
            print(np.sum(attn[j][:prompt_len]))
            lambdas.append(np.sum(attn[j][:prompt_len]))
