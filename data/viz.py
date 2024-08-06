import logomaker
import pandas as pd
import matplotlib.pyplot as plt

# # Sample data: Replace this with your actual sequences
# sequences = [
#     'ACGTACGT',
#     'ACGTTCGT',
#     'ACGTACGG',
#     'ACGTACGC',
#     'ACGTACGT'
# ]

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


import matplotlib.pyplot as plt

# Sample data
data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]

# Create histogram
counts, bins, patches = plt.hist(data, bins=5, edgecolor='black')

# Add title and labels
plt.title('Histogram with Labels')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Add labels to each bin
for count, bin in zip(counts, bins):
    plt.text(bin + (bins[1] - bins[0]) / 2, count, str(int(count)), 
             ha='center', va='bottom')

# Show the plot
plt.show()