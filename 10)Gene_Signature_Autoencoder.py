import pandas as pd
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
import numpy as np

df = pd.read_csv('top_n_representatives.txt',  sep='\t')
X = df.drop('cid', axis=1).values
X = np.asarray(X).astype('float32')


def build_autoencoder(input_dim=978, latent_dim=256):
    # Encoder
    encoder_input = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(978, activation='relu', 
                           activity_regularizer=regularizers.l2(1e-5))(encoder_input)
    encoded = layers.Dropout(0.7)(encoded)  # Dropout layer
    encoded = layers.Dense(489, activation='relu', 
                           activity_regularizer=regularizers.l2(1e-5))(encoded)
    encoded = layers.Dropout(0.5)(encoded)  # Another Dropout layer
    encoded = layers.Dense(489, activation='relu', 
                           activity_regularizer=regularizers.l2(1e-5))(encoded)
    encoded_output = layers.Dense(latent_dim, activation='linear')(encoded)
    
    # Decoder
    decoded = layers.Dense(489, activation='linear', 
                           activity_regularizer=regularizers.l2(1e-5))(encoded_output)
    decoded = layers.Dense(489, activation='relu', 
                           activity_regularizer=regularizers.l2(1e-5))(decoded)
    decoded = layers.Dropout(0.5)(decoded)  # Dropout layer
    decoded = layers.Dense(978, activation='relu', 
                           activity_regularizer=regularizers.l2(1e-5))(decoded)
    decoded = layers.Dropout(0.7)(decoded)  # Another Dropout layer
    decoded_output = layers.Dense(input_dim, activation='linear')(decoded)
    
    # Autoencoder model
    autoencoder = models.Model(encoder_input, decoded_output)
    
    # Encoder model
    encoder = models.Model(encoder_input, encoded_output)
    
    return autoencoder, encoder

autoencoder, encoder = build_autoencoder()
from tensorflow.keras.optimizers.schedules import ExponentialDecay

initial_learning_rate = 0.001
lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=1000, decay_rate=0.90, staircase=True)
optimizer = Adam(learning_rate=lr_schedule)
# Compile the autoencoder
autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')

# Assuming 'fingerprints' is your dataset prepared as a NumPy array

Y= X
history = autoencoder.fit(X, Y, epochs=100, batch_size=256, shuffle=True, validation_split=0.15)

latent_vectors = encoder.predict(X)



# Assuming `latent_vectors` is your numpy array of shape (num_samples, 256) from the encoder
# And `df` is your original DataFrame that includes 'pert_id'

# Convert latent vectors to DataFrame
latent_vectors_df = pd.DataFrame(latent_vectors)

# Check if the number of rows matches
if len(latent_vectors_df) == len(df):
    # Reset index on df to ensure direct alignment
    df_reset = df.reset_index(drop=True)

    # Add 'pert_id' from the original DataFrame to the latent vectors DataFrame
    latent_vectors_df['cid'] = df_reset['cid']
else:
    print("Mismatch in the number of samples between latent vectors and original DataFrame.")


latent_vectors_df.to_csv('sig_lat_vector.txt', sep='\t', index=True)


import matplotlib.pyplot as plt

# Assuming 'history' is your model's training history
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

# Enhancing the Plot
plt.figure(figsize=(6, 6))
plt.plot(epochs, loss, 'b-', linewidth = 2, label='Training Loss')
plt.plot(epochs, val_loss, 'r-', linewidth = 2, label='Validation Loss')
plt.title('Loss Over Epochs for Gene Exp. Autoencoder Training', fontsize=12)

plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Loss (MSE)', fontsize=20)
plt.legend()
plt.yticks([0.3,0.5,0.7,0.9,1.1])
plt.grid(True)
plt.tight_layout()  # Adjusts plot parameters to give specified padding

# Save the figure
plt.savefig('autoencoder_training_validation_loss_transcriptome.svg', format='svg')
