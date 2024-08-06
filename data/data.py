from torch.utils.data import DataLoader, Dataset, random_split
import statistics
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd



class CPP():
    def __init__(
            self,
            sequence: str,
            intensity: float|None = None,
            predicted_intensity: float|None = None,
            ) -> None:
        self.sequence = sequence
        self.intensity = intensity
        self.predicted_intensity = predicted_intensity

    def __len__(self):
        return len(self.sequence)
    
    
    def __getitem__(self, item):
        if item=="intensity":
            return self.intensity
        elif item=="predicted_intensity":
            return self.predicted_intensity
        elif item=="sequence":
            return self.sequence
        
    def __repr__(self):
        return f'sequence:{self.sequence}, intensity:{self.intensity}'


class CPPDataset(Dataset):

    def __init__(
            self,
            cpp_ids,
            attention_mask,
            intensities,
            intensity_ids

    ):
        self.cpp_ids = cpp_ids
        self.attention_mask = attention_mask
        self.intensities = intensities
        self.intensity_ids = intensity_ids

    def __len__(self):
        return len(self.cpp_ids)
    
    def __getitem__(self, idx):
        return self.cpp_ids[idx], self.attention_mask[idx], self.intensities[idx], self.intensity_ids[idx]
    
    

if __name__=="__main__":
    data = pd.read_csv("/home/amirka/CPP/CPPLM/data/cpp.csv").T.to_dict()
    cpps = [CPP(datapoint["sequence"], datapoint['intensity']) for datapoint in data.values()]
    seqs = [cpp['sequence'] for cpp in cpps]
    intensities = [cpp['intensity'] for cpp in cpps]
    print(min(intensities))
    print(max(intensities))
    print(statistics.stdev(intensities))
    # lengths = list(map(lambda x:len(x), seqs))
    # print(max(lengths))
    # cpp_dataset = CPPDataset(cpps)
    # train_dataset, val_dataset, test_dataset = cpp_dataset.split_dataset()
    # train_loader = DataLoader(train_dataset, batch_size=62, shuffle=True)
    # print(train_dataset)