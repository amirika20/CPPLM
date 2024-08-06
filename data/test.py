import pandas as pd
from data import CPP
from statistics import mean, stdev
import matplotlib.pyplot as plt

data = pd.read_csv("/home/amirka/CPP/CPPLM/data/cpp.csv").T.to_dict()
cpps = [CPP(datapoint["sequence"], datapoint['intensity']) for datapoint in data.values()]
intensities = [cpp['intensity'] for cpp in cpps]
sequences = [cpp['sequence'] for cpp in cpps]
lengths = list(map(lambda x:len(x), sequences))

# print("max", max(intensities))
# print("min", min(intensities))
# print("mean", mean(intensities))
# print("stdev", stdev(intensities))
# plt.hist(intensities, bins=10, edgecolor='black')
# plt.title('Distribution of CPPs')
# plt.xlabel('Efficacy relative to PMO')
# plt.ylabel('Frequency')
# plt.show()


print("max", max(lengths))
print("min", min(lengths))
print("mean", mean(lengths))
print("stdev", stdev(lengths))
plt.hist(lengths, bins=10, edgecolor='black')
plt.title('Distribution of CPPs length')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.show()