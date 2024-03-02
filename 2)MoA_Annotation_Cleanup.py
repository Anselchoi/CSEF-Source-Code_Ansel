# Define the path to the original file and the new file
input_file_path = 'compoundinfo.txt'  # Update this path
output_file_path = 'filtered_compoundinfo.txt'  # Update this path

import pandas as pd

# Load the compound information into a pandas DataFrame
compound_df = pd.read_csv('compoundinfo.txt', sep='\t')

# Drop rows with empty values in the 'moa' column
compound_df.dropna(subset=['moa'], inplace=True)

# Drop the last two columns ('inchi_key' and 'compound_aliases')
compound_df = compound_df.iloc[:, :-2]

# Save the cleaned DataFrame to a new text file (optional)
compound_df.to_csv('filtered_compoundinfo.txt', sep='\t', index=False)


print("Filtered file has been created.")
