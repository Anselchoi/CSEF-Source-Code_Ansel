import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Assuming the signatures are saved in 'signatures.txt'
df = pd.read_csv('signatures.txt', sep='\t')

def calculate_cosine_similarities(vectors):
    # Calculate cosine similarity matrix
    return cosine_similarity(vectors)

def find_top_n_references(sim_matrix, n=5):
    # Sum of cosine similarities for each vector
    similarity_sums = sim_matrix.sum(axis=0)
    
    # Indices of the vectors with the top-N highest sums of similarities
    top_n_indices = np.argsort(similarity_sums)[-n:]
    
    return top_n_indices

def compute_weighted_averages(vectors, top_n_indices, sim_matrix):
    top_n_averages = []
    for index in top_n_indices:
        reference_similarities = sim_matrix[index]
        
        # Normalize the similarities to use as weights
        weights = reference_similarities / np.sum(reference_similarities)
        
        # Compute the weighted average for the current top-N reference
        weighted_average = np.average(vectors, axis=0, weights=weights)
        top_n_averages.append(weighted_average)
    
    return top_n_averages

def process_top_n_signatures(df, cid, n=5):
    # Filter the DataFrame for the given cid
    pert_df = df[df['cid'] == cid].drop(['cid'], axis=1)
    
    # Convert DataFrame to numpy array for processing
    vectors = pert_df.to_numpy()
    
    # Calculate cosine similarities
    sim_matrix = calculate_cosine_similarities(vectors)
    
    # Find the top-N reference signatures
    top_n_indices = find_top_n_references(sim_matrix, n=n)
    
    # Compute the weighted averages for top-N signatures
    top_n_weighted_averages = compute_weighted_averages(vectors, top_n_indices, sim_matrix)
    
    return top_n_weighted_averages

# Define the number of top-N representatives you want
N = 5

# Process and store the representatives for each unique cid
unique_cids = df['cid'].unique()
representative_vectors = []
for cid in unique_cids:
    top_n_representatives = process_top_n_signatures(df, cid, n=N)
    for i, rep_vector in enumerate(top_n_representatives):
        representative_vectors.append((f"{cid}", rep_vector))  # Each representative gets a unique identifier

# Prepare the DataFrame for saving
gene_signature_columns = df.columns.drop('cid')
rep_vector_data = [vector for _, vector in representative_vectors]
rep_vector_df = pd.DataFrame(data=rep_vector_data, columns=gene_signature_columns)
rep_vector_df.insert(0, 'cid', [cid for cid, _ in representative_vectors])

# Save the representatives to a new file
rep_vector_df.to_csv('top_n_representatives.txt', sep='\t', index=False)
