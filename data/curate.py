import pandas as pd

complete_dataset = pd.read_csv("data/complete.csv").T.to_dict()
unique_sequences = set()
dataset = []
letters = set()
for data_point in complete_dataset.values():
    sequence = data_point['sequences']
    intensity = data_point['intensity']
    if sequence not in unique_sequences:
        dataset.append({"sequence":sequence, "intensity":intensity})
        unique_sequences.add(sequence)
        letters.update(set([letter for letter in sequence])) 

misc = pd.read_csv("data/misc.csv").T.to_dict()
for data_point in misc.values():
    sequence = data_point['sequence']
    intensity = data_point['intensity']
    if sequence not in unique_sequences:
        dataset.append({"sequence":sequence, "intensity":intensity})
        unique_sequences.add(sequence)
        letters.update(set([letter for letter in sequence]))

df = pd.DataFrame(dataset)
df.to_csv("data/cpp.csv")
print(letters)