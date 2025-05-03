import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the data (replace with your actual path)
file_path = 'API_SP.POP.TOTL_DS2_en_csv_v2_19373.csv'  # Update if the filename is different
df = pd.read_csv(file_path, skiprows=4)  # Skip metadata rows

# Step 2: Extract latest year with available data (example: 2022)
latest_year = '2022'
df = df[['Country Name', latest_year]]
df = df.dropna()

# Step 3: Convert population to millions
df[latest_year] = df[latest_year] / 1_000_000

# Step 4: Plot the histogram
plt.figure(figsize=(10, 6))
bins = [0, 10, 50, 100, 200, 500, 1500]
plt.hist(df[latest_year], bins=bins, edgecolor='black')

plt.title('Distribution of Countries by Population Size (2022)')
plt.xlabel('Population (in millions)')
plt.ylabel('Number of Countries')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
