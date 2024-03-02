import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('updated_moa_compoundinfo.txt', sep='\t')

# Preprocess the 'moa' column to lower case for uniformity
df['moa_processed'] = df['moa'].str.lower()

# Vectorize the 'moa_processed' column
tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
tfidf_matrix = tfidf_vectorizer.fit_transform(df['moa_processed'])

# Cluster the MOAs
k = 150  # Adjust this based on your data analysis
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix)

# Assign the cluster labels to the dataframe
df['cluster'] = clusters

# For each cluster, find the most frequent original 'moa' label
most_frequent_labels = df.groupby('cluster')['moa'].agg(lambda x: pd.Series.mode(x)[0]).reset_index()
most_frequent_labels.rename(columns={'moa': 'representative_moa'}, inplace=True)

#Then we use domain knowledge to assign representative MoA label to each cluster.


'''
#Test Code
# Merge the most frequent labels back to the original dataframe
df = pd.merge(df, most_frequent_labels, how='left', on='cluster')

# Save the dataframe with the reduced MOA categories and their representative labels
df.to_csv('reduced_moa_with_representative_labels.csv', index=False)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Assuming 'tfidf_matrix' is the TF-IDF matrix and 'clusters' are the K-means cluster assignments from your code

# Use t-SNE to reduce dimensionality for visualization
tsne = TSNE(n_components=2, perplexity=40, n_iter=3000, random_state=42)
tfidf_matrix_tsne = tsne.fit_transform(tfidf_matrix.toarray())
import numpy as np

# Calculate the centroids of each cluster in the t-SNE space
centroids_tsne = np.array([tfidf_matrix_tsne[clusters == i].mean(axis=0) for i in range(k)])


plt.figure(figsize=(14, 10))
scatter = plt.scatter(tfidf_matrix_tsne[:, 0], tfidf_matrix_tsne[:, 1], c=clusters, cmap='nipy_spectral', alpha=0.6)
plt.colorbar(scatter, label='Cluster')

# Annotate each cluster centroid with the representative MOA label
for i, centroid in enumerate(centroids_tsne):
    # Retrieve the representative label for the cluster
    representative_label = most_frequent_labels[most_frequent_labels['cluster'] == i]['representative_moa'].values[0]
    plt.text(centroid[0], centroid[1], representative_label, fontsize=6, ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='black', boxstyle='round,pad=0.5'))

plt.title('t-SNE visualization of MOA categories with Representative Labels')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')

# Save the figure as SVG
plt.savefig('tsne_clusters_with_labels.svg', format='svg')
plt.show()
plt.close()


plt.figure(figsize=(14, 10))
scatter = plt.scatter(tfidf_matrix_tsne[:, 0], tfidf_matrix_tsne[:, 1], c=clusters, cmap='nipy_spectral', alpha=0.6)
plt.colorbar(scatter, label='Cluster')

plt.title('t-SNE visualization of MOA categories')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')

# Save the figure as SVG
plt.savefig('tsne_clusters_without_labels.svg', format='svg')
plt.show()
plt.close()
'''
