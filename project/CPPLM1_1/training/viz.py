# Directory configuration
import os
import sys
sys.path.append("/home/amirka/CPP/CPPLM")
sys.path.append("/home/amirka/CPP/CPPLM/project/CPPLM1_0")

# Loading packages
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Loading relative functions and classes
from data.data import CPP, CPPDataset
from data.sampling import *
from tokenization.tokenizer import CPPTokenizer
from tokenization.constants import *


def viz(model):
    tokenizer = CPPTokenizer()
    # data = pd.read_csv("/home/amirka/CPP/CPPLM/data/viz_sample.csv").T.to_dict()
    data = pd.read_csv("/home/amirka/CPP/CPPLM/data/cpp.csv").T.to_dict()
    cpps = [CPP(datapoint["sequence"], datapoint['intensity']/5.017634289122833) for datapoint in data.values()]
    cpps = sorted(cpps, key=lambda x:x.intensity)
    cpp_ids, padding_mask = tokenizer.tokenize_batch(list(map(lambda x:x["sequence"],cpps)))
    # intensities = torch.Tensor(list(map(lambda x:x["intensity"],cpps)))
    output = model(cpp_ids.to("cuda"), padding_mask.to("cuda"))
    print(output.predicted_intensity)
    x = output.embeddings.mean(dim=1).mean(dim=1).unsqueeze(dim=1).cpu().detach().T.numpy()
    plt.imshow(x, cmap='viridis')
    plt.colorbar()
    plt.show()

    # cols = 5  # Number of columns in the grid
    # rows = 1

    # # Create the subplot grid
    # fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    # axes = axes.flatten()

    # for i in range(5):
    #     ax = axes[i]
    #     cax = ax.matshow(output.embeddings[i,:,:].mean(dim=0).unsqueeze(dim=0).cpu().detach().T.numpy(), cmap='viridis')
    #     fig.colorbar(cax, ax=ax)
    #     ax.set_title(f'Tensor {i+1}')

        # print('Target intenesity:', intensities[i]*5.017634289122833)
        # print("Predicted intensity:", output.predicted_intensity[i].cpu()*5.017634289122833)
        # plt.imshow(output.embeddings[i,:len(cpps[i]),:].cpu().detach().T.numpy(), cmap='viridis')
        # plt.colorbar()
        # plt.show()

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    viz(None)