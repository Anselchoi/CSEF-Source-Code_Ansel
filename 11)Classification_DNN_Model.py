import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE, RandomOverSampler
from tensorflow.keras import models, layers, regularizers, optimizers
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import precision_score, recall_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import regularizers

# Load DataFrames
df_cpd = pd.read_csv('lat_vector.txt', sep='\t')
df_sig = pd.read_csv('sig_lat_vector.txt', sep='\t')
df_sig = df_sig.drop(columns=[df_sig.columns[0]], axis=1)
moa_df = pd.read_csv('reduced_moa_with_representative_labels.csv')

# Merge DataFrames on 'pert_id' and 'cid', then merge MOA labels
merged_df = pd.merge(df_cpd, df_sig, left_on='pert_id', right_on='cid').drop(columns=['cid'])
final_df = pd.merge(merged_df, moa_df[['pert_id', 'representative_moa']], on='pert_id', how='left').dropna(subset=['representative_moa'])
final_df['representative_moa'] = final_df['representative_moa'].str.lower().str.replace(' ', '_')

final_df['combined'] = final_df['pert_id'] + ":" + final_df['representative_moa']

# Innitialize One-hot encoder for MOA labels
encoder = OneHotEncoder(sparse=False)
encoder.fit(final_df['representative_moa'].values.reshape(-1, 1))

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import numpy as np

# Prepare Features and Labels
X = final_df.iloc[:, 1:513].values  # Extracting 512 fingerprint columns
y = final_df['combined'].values  # Labels are in 'combined'

# Perform a random initial split: 70% training, 30% pool
X_train_initial, pool_X, y_train_initial, pool_y = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Cosine Similarity Filtering to prevent data leakage
# Placeholder for indices to keep in the pool
indices_to_keep = []
for i, vector in enumerate(pool_X):
    # Calculate cosine similarities between the vector and all in the initial training set
    similarities = cosine_similarity([vector], X_train_initial)[0]
    
    # Sort the similarities and take the top 10, excluding the highest one since it's the vector itself
    top_n_similarities = np.partition(similarities, -6)[-6:-1]  # Use partition for efficiency
    
    # Calculate the average of the top 6 similarities
    avg_top_n_sim = np.mean(top_n_similarities)
    
    # Define a threshold for the average similarity (e.g., 0.9)
    # Keep the vector if its average top 10 similarity is less than the threshold
    if avg_top_n_sim <= 0.925:
        indices_to_keep.append(i)

# Filter the pool based on indices to keep
filtered_pool_X = pool_X[indices_to_keep]
filtered_pool_y = pool_y[indices_to_keep]

# Further Split the remaining 30% Into Validation and Test Sets
X_val, X_test, y_val, y_test = train_test_split(
    filtered_pool_X, filtered_pool_y, test_size=0.5, random_state=42
)




# Handling Imbalanced Data with RandomOverSampler for training set
ros = RandomOverSampler(sampling_strategy = 'auto')
X_train, y_train = ros.fit_resample(X_train_initial, y_train_initial)

# Initialize SMOTE
smote = SMOTE(sampling_strategy='auto')  # Adjust the strategy as needed
X_train, y_train = smote.fit_resample(X_train, y_train)

# Split the 'combined' column back into 'pert_id' and 'representative_moa'
y_train = pd.DataFrame(y_train, columns=['combined'])
split_df = y_train['combined'].str.split(":", n=1, expand=True)
y_train['pert_id'] = split_df[0]
y_train['representative_moa'] = split_df[1]
y_train = y_train.drop(['combined'], axis=1)

y_val = pd.DataFrame(y_val, columns=['combined'])
split_df = y_val['combined'].str.split(":", n=1, expand=True)
y_val['pert_id'] = split_df[0]
y_val['representative_moa'] = split_df[1]
y_val = y_val.drop(['combined'], axis=1)

y_test = pd.DataFrame(y_test, columns=['combined'])
split_df = y_test['combined'].str.split(":", n=1, expand=True)
y_test['pert_id'] = split_df[0]
y_test['representative_moa'] = split_df[1]
y_test = y_test.drop(['combined'], axis=1)

# Save 'pert_id' for later referencing
y_val_pert_ids = y_test['pert_id']

# Drop the 'pert_id' column as it's no longer needed
y_train = y_train.drop(['pert_id'], axis=1)
y_val = y_val.drop(['pert_id'], axis=1)
y_test = y_test.drop(['pert_id'], axis=1)
# Now, resampled_df contains the separated 'pert_id' and 'representative_moa' columns

# Encode text labels to one-hot
y_train = encoder.transform(y_train.values.reshape(-1, 1))
y_val = encoder.transform(y_val.values.reshape(-1, 1))
y_test = encoder.transform(y_test.values.reshape(-1, 1))




# Model Building
def build_model(input_dim, num_classes):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        Dropout(0.5),
        Dense(255, activation='relu', activity_regularizer=regularizers.l2(1e-5)),
        Dropout(0.5),
        Dense(255, activation='relu', activity_regularizer=regularizers.l2(1e-5)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
model = build_model(X_train.shape[1], y_train.shape[1])


# Custom Callback for Precision, Recall, and F1
class Metrics(Callback):
    def __init__(self, train_data, validation_data):
        super(Metrics, self).__init__()
        self.train_data = train_data
        self.validation_data = validation_data
        # Initialize lists to store metrics
        self.train_precisions = []
        self.val_precisions = []
        self.train_recalls = []
        self.val_recalls = []
        self.train_f1s = []
        self.val_f1s = []

    def on_epoch_end(self, epoch, logs=None):
        # Predictions for training data
        train_predictions = (self.model.predict(self.train_data[0]) > 0.5).astype("int32")
        # Compute metrics for training data
        self.train_precisions.append(precision_score(self.train_data[1], train_predictions, average='macro', zero_division=0))
        self.train_recalls.append(recall_score(self.train_data[1], train_predictions, average='macro'))
        self.train_f1s.append(f1_score(self.train_data[1], train_predictions, average='macro'))
        
        # Predictions for validation data
        val_predictions = (self.model.predict(self.validation_data[0]) > 0.5).astype("int32")
        # Compute metrics for validation data
        self.val_precisions.append(precision_score(self.validation_data[1], val_predictions, average='macro', zero_division=0))
        self.val_recalls.append(recall_score(self.validation_data[1], val_predictions, average='macro'))
        self.val_f1s.append(f1_score(self.validation_data[1], val_predictions, average='macro'))

        print(f'Epoch {epoch+1}: Training Precision: {self.train_precisions[-1]}, Validation Precision: {self.val_precisions[-1]}')
        print(f'Epoch {epoch+1}: Training Recall: {self.train_recalls[-1]}, Validation Recall: {self.val_recalls[-1]}')
        print(f'Epoch {epoch+1}: Training F1: {self.train_f1s[-1]}, Validation F1: {self.val_f1s[-1]}')
metrics_callback = Metrics(train_data=(X_train, y_train), validation_data=(X_val, y_val))

# Model Training
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=512, callbacks=[metrics_callback])





#Plotting Loss, Accuracy vs Epochs
# Assuming 'history' is your model's training history
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(loss) + 1)


# Assuming 'epochs', 'loss', and 'val_loss' are defined
plt.figure(figsize=(6, 6))
plt.plot(epochs, loss, 'b-', label='Training Loss', linewidth=2)  # Increase line width here
plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)  # And here
plt.title('Loss Over Epochs for MoA Model Training', fontsize=14)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Loss (Categorical Crossentropy)', fontsize=20)
plt.legend()
plt.grid(True)
plt.tight_layout()  # Adjusts plot parameters to give specified padding
plt.savefig('training_validation_loss_model_3.svg', format='svg')
plt.show()
# Save the figure
plt.close()

plt.figure(figsize=(6, 6))
plt.plot(epochs, acc, 'b-', label='Training Acc', linewidth=2)  # Increase line width here
plt.plot(epochs, val_acc, 'r-', label='Validation Acc', linewidth=2)  # And here
plt.title('Accuracy Over Epochs for MoA Model Training', fontsize=14)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.legend()
plt.grid(True)
plt.tight_layout()  # Adjusts plot parameters to give specified padding
plt.savefig('training_validation_acc_model_3.svg', format='svg')
plt.show()
# Save the figure
plt.close()



#Plotting Metrics vs Epochs
epochs = range(1, len(metrics_callback.train_precisions) + 1)

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(epochs, metrics_callback.train_precisions, label='Training Precision')
plt.plot(epochs, metrics_callback.val_precisions, label='Validation Precision')
plt.title('Training and Validation Precision Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(epochs, metrics_callback.train_recalls, label='Training Recall')
plt.plot(epochs, metrics_callback.val_recalls, label='Validation Recall')
plt.title('Training and Validation Recall Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(epochs, metrics_callback.train_f1s, label='Training F1 Score')
plt.plot(epochs, metrics_callback.val_f1s, label='Validation F1 Score')
plt.title('Training and Validation F1 Score Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('metrics.svg', format='svg')
plt.show()
plt.close()




#Plotting P-R Curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# After training, predict on the test set
# Model's predicted probabilities
y_true_binarized = y_test
y_scores = model.predict(X_test)  

# Compute micro-average Precision-Recall curve and area
precision, recall, _ = precision_recall_curve(y_true_binarized.ravel(), y_scores.ravel())
average_precision = average_precision_score(y_true_binarized, y_scores, average="micro")

# Plotting the Averaged Precision-Recall Curve
plt.figure(figsize=(6, 6))
plt.step(recall, precision, where='post', label=f'Precision-Recall (area = {average_precision:0.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Test Set Precision-Recall curve')
plt.grid(True)
plt.legend(loc="best")

plt.savefig('test_precisionvsrecall_model.svg', format='svg')
plt.show()
plt.close()


'''
#Test Code
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# Assuming y_test are your true labels encoded as integers
# And model.predict(X_val) returns the predicted probabilities for each class
y_pred_proba = model.predict(X_test)

# Binarize the output labels for multi-class
y_val_binarized = y_test

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(y_test.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_val_binarized[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure(figsize=(6, 6))

for i in range(y_test.shape[1]):
    plt.plot(fpr[i], tpr[i], lw=0.9)

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Test Set Receiver Operating Characteristic for Each Class For the Model')
plt.grid(True)
plt.savefig('each_class_ROC_model.svg', format='svg')

plt.show()



# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(y_test.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_val_binarized[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(y_test.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_val_binarized[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_val_binarized.ravel(), y_pred_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(y_test.shape[1])]))

# Then interpolate all ROC curves at these points
mean_tpr = np.zeros_like(all_fpr)
for i in range(y_test.shape[1]):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])



# Finally, average it and compute AUC
mean_tpr /= y_test.shape[1]
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

### Plotting
plt.figure(figsize=(6, 6))
plt.plot(fpr["macro"], tpr["macro"],
         label=f'Class-average ROC curve (area = {roc_auc["macro"]:.2f})',
         color='navy', linestyle=':', linewidth=4)


plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Test Set Class-average ROC curves')
plt.grid(True)
plt.savefig('avg_class_ROC_model.svg', format='svg')
plt.legend(loc="lower right")
plt.show()
'''

# Predict the probabilities for the test set
y_pred_proba = model.predict(X_test)

# Assuming the encoder fitted as shown previously
moa_labels = encoder.inverse_transform(np.eye(y_pred_proba.shape[1]))  # Get MOA labels
# Assuming y_pred_proba and moa_labels are already defined
# y_pred_proba = model.predict(X_val)
# moa_labels = encoder.classes_ (if you used OneHotEncoder and fit it earlier)

def get_top_n_predictions(probas, labels, n=10):
    top_predictions = []
    for proba in probas:
        top_indices = np.argsort(proba)[-n:][::-1]  # Get indices of top N predictions
        top_moa = labels[top_indices]  # Map indices to MOA labels
        top_scores = proba[top_indices]  # Get corresponding scores
        top_predictions.append(list(zip(top_moa, top_scores)))
    return top_predictions

top_n_predictions = get_top_n_predictions(y_pred_proba, moa_labels, n=10)
# Prepare the data for the DataFrame
data_for_df = []
for i, preds in enumerate(top_n_predictions):
    pert_id = y_val_pert_ids.iloc[i]  # Assuming y_val_pert_ids is a Series with pert_id for validation examples
    for rank, (moa, score) in enumerate(preds, start=1):
        data_for_df.append({
            "pert_id": pert_id,
            "rank": rank,
            "moa": moa,
            "score": score
        })
        
# Create the DataFrame
predictions_df = pd.DataFrame(data_for_df)
predictions_df.to_csv("top_10_moa_predictions.csv", index=False)

