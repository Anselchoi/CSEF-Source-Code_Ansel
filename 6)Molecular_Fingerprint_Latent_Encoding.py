import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

# Load your dataset
df = pd.read_csv('reduced_moa_with_representative_labels.csv')

# Filter out rows with NaN SMILES
df_filtered = df.dropna(subset=['canonical_smiles'])

# Define a function to convert SMILES to fingerprints
def smiles_to_fingerprint(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits).ToList() # Convert to list for easier handling
    else:
        print('error')
        return [0] * n_bits  # Return a zero vector if the molecule couldn't be parsed
    

# Convert SMILES to fingerprints
df_filtered['fingerprint'] = df_filtered['canonical_smiles'].apply(lambda x: smiles_to_fingerprint(x))

# Since we're operating on a copy of the DataFrame, it's safe to use `.apply()` here,
# but be mindful of SettingWithCopyWarning in pandas.

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam

def build_autoencoder(input_dim=2048, latent_dim=256):
    # Encoder
    encoder_input = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(1024, activation='relu', 
                           activity_regularizer=regularizers.l2(1e-5))(encoder_input)
    encoded = layers.Dropout(0.5)(encoded)  # Dropout layer
    encoded = layers.Dense(512, activation='relu', 
                           activity_regularizer=regularizers.l2(1e-5))(encoded)
    encoded = layers.Dropout(0.5)(encoded)  # Another Dropout layer
    encoded_output = layers.Dense(latent_dim, activation='relu')(encoded)
    
    # Decoder
    decoded = layers.Dense(512, activation='relu', 
                           activity_regularizer=regularizers.l2(1e-5))(encoded_output)
    decoded = layers.Dropout(0.5)(decoded)  # Dropout layer
    decoded = layers.Dense(1024, activation='relu', 
                           activity_regularizer=regularizers.l2(1e-5))(decoded)
    decoded = layers.Dropout(0.5)(decoded)  # Another Dropout layer
    decoded_output = layers.Dense(input_dim, activation='sigmoid')(decoded)
    
    # Autoencoder model
    autoencoder = models.Model(encoder_input, decoded_output)
    
    # Encoder model
    encoder = models.Model(encoder_input, encoded_output)
    
    return autoencoder, encoder

autoencoder, encoder = build_autoencoder()

# Compile the autoencoder
autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy')

# Assuming 'fingerprints' is your dataset prepared as a NumPy array
X = np.stack(df_filtered['fingerprint'].values)
Y= np.stack(df_filtered['fingerprint'].values)
history = autoencoder.fit(X, Y, epochs=100, batch_size=256, shuffle=True, validation_split=0.15)

latent_vectors = encoder.predict(X)



# Assuming `latent_vectors` is your numpy array of shape (num_samples, 256) from the encoder
# And `df` is your original DataFrame that includes 'pert_id'

# Convert latent vectors to DataFrame
latent_vectors_df = pd.DataFrame(latent_vectors)

# Check if the number of rows matches
if len(latent_vectors_df) == len(df_filtered):
    # Reset index on df to ensure direct alignment
    df_reset = df_filtered.reset_index(drop=True)

    # Add 'pert_id' from the original DataFrame to the latent vectors DataFrame
    latent_vectors_df['pert_id'] = df_reset['pert_id']
else:
    print("Mismatch in the number of samples between latent vectors and original DataFrame.")

# If you wish to have 'pert_id' as the first column, you can reorder the DataFrame columns
cols = ['pert_id'] + [col for col in latent_vectors_df if col != 'pert_id']
latent_vectors_df = latent_vectors_df[cols]


latent_vectors_df.to_csv('lat_vector.txt', sep='\t', index=False)


import matplotlib.pyplot as plt

# Assuming 'history' is your model's training history
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

# Enhancing the Plot
plt.figure(figsize=(6, 6))
plt.plot(epochs, loss, 'b-', linewidth = 2, label='Training Loss')
plt.plot(epochs, val_loss, 'r-', linewidth = 2, label='Validation Loss')
plt.title('Loss Over Epochs for Fingerprint Autoencoder Training', fontsize=14)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Loss (Binary Crossentropy)', fontsize=20)
plt.legend()
plt.grid(True)
plt.tight_layout()  # Adjusts plot parameters to give specified padding

# Save the figure
plt.savefig('autoencoder_training_validation_loss_fingerprint.svg', format='svg')

