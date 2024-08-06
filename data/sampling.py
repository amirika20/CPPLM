import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
import pandas as pd
from data.data import CPP, CPPDataset
import matplotlib.pyplot as plt
import logomaker
from functools import reduce

def stratified_split(cpps, n_bins=9, train_size=0.7, val_size=0.2, test_size=0.1, random_seed=40):

    intensities = [cpp['intensity'] for cpp in cpps]
    # Create bins
    # bins = np.linspace(min(intensities), max(intensities), n_bins+1)
    bins = np.linspace(-0.001, max(intensities), n_bins+1)

    bin_indices = np.digitize(intensities, bins, right=True)

    # Create a dictionary to hold the binned objects
    binned_cpps = {i: [] for i in range(0, len(bins))}
    
    # Group objects by their bin indices
    for cpp, bin_index in zip(cpps, bin_indices):
        binned_cpps[bin_index].append(cpp)

    # Initialize the splits
    train_cpps, val_cpps, test_cpps = [], [], []
    
    # Stratified sampling
    for i in range(n_bins):
        bin_data = binned_cpps[i]
        if len(bin_data)<3:
            train_bin = bin_data
            val_bin = []
            test_bin = []
        else:
            train_bin, temp_bin = train_test_split(bin_data, test_size=(1-train_size), random_state=random_seed)
            val_bin, test_bin = train_test_split(temp_bin, test_size=test_size/(test_size + val_size), random_state=random_seed)
        
        train_cpps.extend(train_bin)
        val_cpps.extend(val_bin)
        test_cpps.extend(test_bin)
    
    return train_cpps, val_cpps, test_cpps


def sampling(cpps, n_bins=9):
    is_valid = False
    while not is_valid:
        train_cpps, val_cpps, test_cpps = stratified_split(cpps, n_bins=n_bins)

        train_intensities = [cpp['intensity'] for cpp in train_cpps]
        val_intensities = [cpp['intensity'] for cpp in val_cpps]
        test_intensities = [cpp['intensity'] for cpp in test_cpps]

        train_val_ks_stat, train_val_p_value = ks_2samp(train_intensities, val_intensities)
        train_test_ks_stat, train_test_p_value = ks_2samp(train_intensities, test_intensities)
        val_test_ks_stat, val_test_p_value = ks_2samp(val_intensities, test_intensities)

        is_valid = train_val_p_value >= 0.2 and train_test_p_value >= 0.2 and val_test_p_value >= 0.2

    print(f"Train vs. Validation KS Statistic: {train_val_ks_stat}, P-Value: {train_val_p_value}")
    print(f"Train vs. Test KS Statistic: {train_test_ks_stat}, P-Value: {train_test_p_value}")
    print(f"Validation vs. Test KS Statistic: {val_test_ks_stat}, P-Value: {val_test_p_value}")

    # # Interpretation
    if is_valid:
        print("The distributions are similar across training, validation, and test sets.")
    else:
        print("There is a significant difference in the distributions.")

    return train_cpps, val_cpps, test_cpps



if __name__ == "__main__":
    data = pd.read_csv("/home/amirka/CPP/CPPLM/data/cpp.csv").T.to_dict()
    cpps = [CPP(datapoint["sequence"], datapoint['intensity']) for datapoint in data.values()]
    intensities = [cpp['intensity'] for cpp in cpps]
    n_bins = 18
    # Create bins
    bins = np.linspace(-0.0001, max(intensities), n_bins+1)
    
    bin_indices = np.digitize(intensities, bins, right=True)

    # Create a dictionary to hold the binned objects
    binned_cpps = {i: [] for i in range(0, len(bins))}
    
    # Group objects by their bin indices
    for cpp, bin_index in zip(cpps, bin_indices):
        binned_cpps[bin_index].append(cpp['sequence'])
    del binned_cpps[0]
    avg_length = {}
    for bin in binned_cpps.keys():
        seqs = binned_cpps[bin]
        lengths = list(map(lambda x:len(x), seqs))
        avg = sum(lengths)/len(lengths)
        avg_length[bin] = avg
    

    # intensities = [datapoint['intensity'] for datapoint in data.values()]
    counts, bins, patches = plt.hist(intensities, bins=n_bins, edgecolor='black')


    # Add title and labels
    plt.title('Distribution of CPPs')
    plt.xlabel('Efficacy relative to PMO')
    plt.ylabel('Frequency')

    for count, bin_edge, length in zip(counts, bins, list(avg_length.values())):
        bin_center = bin_edge + (bins[1] - bins[0]) / 2
        label = f'Count: {int(count)}\nAvg Len: {int(length)}'
        plt.text(bin_center, count, label, ha='center', va='bottom')

    # Show the plot
    # plt.savefig('/home/amirka/CPP/CPPLM/figures/CPPs_5bins.png')
    plt.show()





    # sequences = [datapoint["sequence"] for datapoint in data.values()]

    # # Convert sequences to a matrix format
    # count_matrix = logomaker.alignment_to_matrix(sequences)

    # # Create a DataFrame
    # df = pd.DataFrame(count_matrix)

    # # Create the logo
    # logo = logomaker.Logo(df)

    # # Add title and labels
    # logo.ax.set_title('Sequence Logo')
    # logo.ax.set_xlabel('Position')
    # logo.ax.set_ylabel('Frequency')

    # # Show the plot
    # plt.show()