import pandas as pd

# Load the dataset
df = pd.read_csv('canonical_smiles_compoundinfo_no_duplicates.txt', sep='\t')

# Define patterns for replacement
replacement_patterns = {
    r'\bagonist\b|\bactivator\b|\bstimulator\b|\bstimulant\b': 'activator',  # Replace 'agonist' or 'activator' with 'activator'
    r'\binhibitor\b|\bantagonist\b|\bblocker\b': 'inhibitor'  # Replace 'inhibitor', 'antagonist', or 'blocker' with 'inhibitor'
}

# Apply replacements
for pattern, replacement in replacement_patterns.items():
    df['moa'] = df['moa'].str.replace(pattern, replacement, case=False, regex=True)

import pandas as pd

# Count the occurrences of each MoA
moa_counts = df['moa'].value_counts()

# Filter out MoAs that occur only once
df_filtered = df[df['moa'].map(moa_counts) > 1]

print(df_filtered['moa'].nunique())
# Save the updated dataframe to a new file
df_filtered.to_csv('updated_moa_compoundinfo.txt', sep='\t', index=False)
