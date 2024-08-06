import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table

# Sample data
data = {
    'Month': ['January', 'February', 'March', 'April', 'May'],
    'Sales': [200, 300, 400, 500, 600]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Plotting the table
fig, ax = plt.subplots(figsize=(8, 4))  # set size frame
ax.axis('tight')
ax.axis('off')
tbl = table(ax, df, loc='center', cellLoc='center', colWidths=[0.2] * len(df.columns))

# Save the figure
plt.savefig('sales_table.png')

plt.show()
