import pkg_resources

# Print version of cmapPy being used in current conda environment 
print(pkg_resources.get_distribution("cmapPy").version)


import pandas as pd

gene_info = pd.read_csv("geneinfo.txt", sep="\t")
print(gene_info.columns)
landmark_gene_row_ids = gene_info["pr_gene_id"][gene_info["pr_is_lm"] == 1].tolist()
landmark_gene_row_ids = [str(element) for element in landmark_gene_row_ids]


from cmapPy.pandasGEXpress.parse import parse

landmark_only_gctoo = parse("LINCS.gctx", rid = landmark_gene_row_ids)
df = landmark_only_gctoo.data_df


# Path to the text file
text_file_path = 'matched_siginfo.txt'
# Read the text file
siginfo_df = pd.read_csv(text_file_path, sep='\t')
# Extract sig_id values
sig_ids = siginfo_df['sig_id'].tolist()


# Select columns from the DataFrame
try:
    selected_columns = df[sig_ids]
except KeyError as e:
    print(f"Column not found: {e}")



#Create a mapping from sig_id to pert_id
sig_to_pert_mapping = pd.Series(siginfo_df.pert_id.values,index=siginfo_df.sig_id).to_dict()

# Filter out valid sig_ids that are also column names in the DataFrame
valid_sig_ids = [sig_id for sig_id in sig_to_pert_mapping if sig_id in selected_columns.columns]

# Create a new mapping for renaming, focusing only on valid columns present in df
rename_mapping = {sig_id: sig_to_pert_mapping[sig_id] for sig_id in valid_sig_ids}

# Rename the columns in the DataFrame
selected_columns_df = selected_columns.rename(columns=rename_mapping)

print("DataFrame with columns renamed from sig_id to pert_id:")
print(selected_columns_df.head())
trans = selected_columns_df.T
trans.to_csv('signatures.txt', sep='\t', index=True)
