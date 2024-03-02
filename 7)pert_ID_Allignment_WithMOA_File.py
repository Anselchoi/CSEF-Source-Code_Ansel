import pandas as pd

# Assuming the reduced MoA file is saved as 'reduced_moa.csv'
moa_df = pd.read_csv('lat_vector.txt', sep='\t')

# Load the gene signatures file
siginfo_df = pd.read_csv('siginfo.txt', sep='\t')

# Initialize an empty list to store rows where 'pert_id' is found in the MoA file
matched_rows = []

# Iterate over each row in siginfo_df to check if its 'pert_id' is in moa_df
for index, row in siginfo_df.iterrows():
    pert_id = row['pert_id']
    if pert_id in moa_df['pert_id'].values:
        matched_rows.append(row)
    else:
        print(f"No Match for {pert_id}")

# Convert the matched rows to a DataFrame
matched_df = pd.DataFrame(matched_rows)

# Check if matched_df is empty
if matched_df.empty:
    print("No matching instances found.")
else:
    # Save the matched rows to a new file
    matched_df.to_csv('matched_siginfo.txt', sep='\t', index=False)