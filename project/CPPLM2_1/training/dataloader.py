# Directory configuration
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Loading packages
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Loading relative functions and classes
from data.data import CPP, CPPDataset
from data.sampling import *
from tokenization.tokenizer import CPPTokenizer
from tokenization.constants import *

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
        # std = pd.read_csv(self.data_dir)["intensity"].std()
        cpps = [CPP(datapoint["sequence"], datapoint['intensity']) for datapoint in data.values()]
        self.data = cpps

        # Tokenizer
        self.tokenizer = CPPTokenizer()

    def encode(
            self,
            cpps: list[CPP],
            ):
        cpp_ids, padding_mask = self.tokenizer.tokenize_batch(list(map(lambda x:x["sequence"],cpps)))
        intensities = torch.Tensor(list(map(lambda x:x["intensity"],cpps)))
        bin_edges = torch.tensor(INTENSITY_BOUNDARIES)
        intensity_ids = torch.bucketize(intensities, bin_edges).long()
        return CPPDataset(cpp_ids, padding_mask, intensities, intensity_ids)

    def prep_data(
            self,
            batch_size: int=32,
            ):
        train_cpps, val_cpps, test_cpps = sampling(self.data)
        train_dataset = self.encode(train_cpps)
        val_dataset = self.encode(val_cpps)
        test_dataset = self.encode(test_cpps)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)
        print(f'Training size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}')
        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    dataset = CPPDataloader()
    train_loader, val_loader, test_loader = dataset.prep_data()
    for point in train_loader:
        print(point[0][2])
        break
    