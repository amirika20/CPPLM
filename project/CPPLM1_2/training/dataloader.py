# Directory configuration
import os
import sys
sys.path.append("/home/amirka/CPP/CPPLM")

# Loading packages
import esm.pretrained
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import esm
from tqdm import tqdm

# Loading relative functions and classes
from data.data import CPP
from data.sampling import *

class CPPESMDataset(Dataset):

    def __init__(
            self,
            CPP_ESM_embedding,
            intensities,
            padding_mask,
    ):
        self.cpp_embedding = CPP_ESM_embedding
        self.intensities = intensities
        self.padding_mask = padding_mask

    def __len__(self):
        return len(self.intensities)
    
    def __getitem__(self, idx):
        return self.cpp_embedding[idx], self.intensities[idx], self.padding_mask[idx]

class CPPDataloader():

    def __init__(
            self,
            data_dir: str|None=None
    ):
        
        # Data Directory
        self.data_dir = data_dir
        if data_dir:
            self.data_dir = data_dir
        else:
            self.data_dir = "/home/amirka/CPP/CPPLM/data/cpp.csv"

        # Loading data
        data = pd.read_csv(self.data_dir).T.to_dict()
        std = pd.read_csv(self.data_dir)["intensity"].std()
        cpps = [CPP(datapoint["sequence"], datapoint['intensity']/std) for datapoint in data.values()]
        self.data = cpps

        # Tokenizer
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        self.esm_model = model.to("cuda")


    def encode(
            self,
            cpps: list[CPP],
            ):
        seqs = list(map(lambda x:(x["intensity"], x["sequence"].replace("2","").replace("3","")),cpps))
        batch_labels, batch_strs, batch_tokens = self.batch_converter(seqs)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        token_representations = []
        with torch.no_grad():
            for seq_num in tqdm(range(len(seqs)), desc="ESM Encoding"):
                results = self.esm_model(batch_tokens[seq_num:seq_num+1,:].to("cuda"), repr_layers=[36])
                token_representations.append(results["representations"][36])
        token_representations = torch.cat(token_representations, dim=0)
        sequence_representations = []
        # for i, tokens_len in enumerate(batch_lens):
        #     sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0).float())
        padding_mask = (batch_tokens!=1)*1
        return CPPESMDataset(token_representations, batch_labels, padding_mask)

    def prep_data(
            self,
            batch_size: int=32,
            n_bins: int=9,
            ):
        train_cpps, val_cpps, test_cpps = sampling(self.data, n_bins=n_bins)
        train_dataset = self.encode(train_cpps)
        val_dataset = self.encode(val_cpps)
        test_dataset = self.encode(test_cpps)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)
        print(f'Training size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}')
        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    dataset = CPPDataloader()
    dataset.prep_data()
    # train_loader, val_loader, test_loader = dataset.prep_data()
    # for point in train_loader:
    #     print(point[0][2])
    #     break
    