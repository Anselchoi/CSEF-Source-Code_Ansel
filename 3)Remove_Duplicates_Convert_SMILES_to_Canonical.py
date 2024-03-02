from rdkit import Chem
import pandas as pd

# Load the dataset
df = pd.read_csv('filtered_compoundinfo.txt', sep='\t')

# Remove duplicates based on 'pert_id', keeping the first occurrence
df = df.drop_duplicates(subset='pert_id', keep='first')

# Ensure the SMILES column is treated as a string and handle missing/non-string values
df['canonical_smiles'] = df['canonical_smiles'].astype(str)

# Define a function to convert SMILES to canonical SMILES, with error handling
def to_canonical(smiles):
    try:
        # Attempt to create a molecule object from the SMILES string
        mol = Chem.MolFromSmiles(smiles)
        if mol:  # Check if the molecule object is created successfully
            return Chem.MolToSmiles(mol, canonical=True)
    except Exception as e:
        print(f"Error converting SMILES: {smiles} - {e}")
    return None  # Return None or some placeholder for invalid/failed conversions

# Apply the conversion function to the SMILES column
df['canonical_smiles'] = df['canonical_smiles'].apply(lambda x: to_canonical(x) if x != 'nan' else x)

# Optionally, filter out rows where conversion was not successful if you don't want to keep them
df = df[df['canonical_smiles'].notnull()]

# Save the updated dataframe to a new file
df.to_csv('canonical_smiles_compoundinfo_no_duplicates.txt', sep='\t', index=False)
