# Directory configuration
import os
import sys
sys.path.append("/home/amirka/CPP/CPPLM")
sys.path.append("/home/amirka/CPP/CPPLM/project/CPPLM1_0")

# Loading packages
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Loading relative functions and classes
from data.data import CPP, CPPDataset
from data.sampling import *
from tokenization.tokenizer import CPPTokenizer
from tokenization.constants import *


def mach(model):
    tokenizer = CPPTokenizer()
    data = pd.read_csv("/home/amirka/CPP/CPPLM/data/mach.csv").T.to_dict()
    cpps = [CPP(datapoint["sequence"], datapoint['intensity']/5.017634289122833) for datapoint in data.values()]
    cpp_ids, padding_mask = tokenizer.tokenize_batch(list(map(lambda x:x["sequence"],cpps)))
    exp_intensities = [datapoint['intensity'] for datapoint in data.values()]
    output = model(cpp_ids.to("cuda"), padding_mask.to("cuda"))
    pred_int = output.predicted_intensity
    print(exp_intensities)
    print(pred_int.cpu().detach()*5.017634289122833)




if __name__ == "__main__":
    viz(None)