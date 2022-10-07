import torch
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    attn_dir = "attn_weight/hubert_asr_pre50_asrinit_ldim512_projdim64_short"
    attn_pths = [os.path.join(attn_dir, f"layer{i}.pt") for i in range(12)]

    for i in range(12):
        print(i)
        attn = torch.load(attn_pths[i]) # (1, 168, 168)
        attn = attn.squeeze(0).numpy()
        plt.imshow(attn, cmap='rainbow', interpolation='nearest')
        plt.colorbar()
        plt.title(f"layer {i}")
        plt.savefig(os.path.join(attn_dir, f"attn{i}.png"))
        plt.cla()
        plt.clf()
    fig, ax = plt.subplots(3, 4)
    for y in range(3):
        for x in range(4):
            print(i)
            i = 4 * y + x
            attn = torch.load(attn_pths[i])
            attn = attn.squeeze(0).numpy()
            img = ax[y][x].imshow(attn, cmap='rainbow', interpolation='nearest')
            ax[y][x].set_title(f"layer {i}")
            ax[y][x].axis('off')
    plt.colorbar(img, ax=ax)
    plt.savefig(os.path.join(attn_dir, "attn.png"))

