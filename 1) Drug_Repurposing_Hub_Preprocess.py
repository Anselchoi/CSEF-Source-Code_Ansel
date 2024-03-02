import pandas as pd

# Load the repurposing samples and drugs text files into pandas DataFrames
samples_df = pd.read_csv('repurposing_samples.txt', sep='\t')
drugs_df = pd.read_csv('repurposing_drugs.txt', sep='\t')

# Modify the 'broad_id' in samples_df to only include the part up to the second hyphen
samples_df['broad_id'] = samples_df['broad_id'].apply(lambda x: '-'.join(x.split('-')[:2]))

# Merge the two DataFrames on the 'pert_iname' column to find corresponding 'broad_id' and 'smiles' for each 'pert_iname' in the drugs text
merged_df = pd.merge(drugs_df, samples_df[['pert_iname', 'broad_id', 'smiles']], on='pert_iname', how='left')

# Select only the 'pert_iname', 'broad_id', and 'smiles' columns (if you want other columns from the drugs file, adjust this accordingly)
final_df = merged_df[['pert_iname', 'broad_id', 'smiles']]

# Save the result to a new text file
merged_df.to_csv('matched_pert_iname_broad_id_smiles.txt', sep='\t', index=False)
