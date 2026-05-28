"""
Project: Network Intrusion Detection System (NIDS)
Description: An AI-driven cybersecurity analytics platform designed to ingest high-dimensional 
             network traffic data, execute data preparation pipelines, and classify network packets 
             as 'Normal' or 'Attack' using a Random Forest Classifier.
Dataset: NSL-KDD (KDDTrain+.txt and KDDTest+.txt)
Primary Stack: Python, Pandas, NumPy, Scikit-Learn
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
# Set to True to validate against the official test set (unknown/novel attacks).
# Set to False to perform an 80/20 train-test split on the training file (known attacks).
use_kdd_test_file = False

# Official feature names for the NSL-KDD dataset attributes
col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
             "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
             "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
             "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
             "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
             "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
             "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
             "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
             "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"]

# ==========================================
# DATA INGESTION
# ==========================================
try:
    # Load dataset files without headers and map the explicit column schema
    train_df = pd.read_csv("KDDTrain+.txt", header=None, names=col_names)
    test_df = pd.read_csv("KDDTest+.txt", header=None, names=col_names)
except FileNotFoundError:
    print("Error: Dataset files not found.")
    print("Please download 'KDDTrain+.txt' and 'KDDTest+.txt' and place them in the same directory.")
    exit()

print("Data loaded successfully.")
print("Preprocessing data...")

# ==========================================
# DATA PREPARATION & PIPELINE ROUTINES
# ==========================================
# Determine validation strategy based on the initial configuration flag
if use_kdd_test_file:
    print("Mode: Testing against KDDTest+.txt (unknown attacks)")
    # Concatenate dataframes to ensure unified one-hot encoding feature spaces
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    split_index = len(train_df)
else:
    print("Mode: Splitting KDDTrain+.txt (known attacks)")
    combined_df = train_df.copy()
    split_index = 0 

# Map multiclass attack labels to a strict binary classification problem: Normal (0) vs Attack (1)
combined_df['label'] = combined_df['label'].apply(lambda x: 0 if x == 'normal' else 1)

# Drop 'difficulty' metric as it is a dataset metadata attribute, not a network traffic feature
combined_df = combined_df.drop(columns=['difficulty'])

# Segregate categorical and numerical features for targeted data preparation
categorical_cols = ['protocol_type', 'service', 'flag']
numerical_cols = [col for col in combined_df.columns if col not in categorical_cols + ['label']]

# Encode categorical variables using standard One-Hot Encoding (binary vectors)
print("Applying one-hot encoding...")
combined_df = pd.get_dummies(combined_df, columns=categorical_cols, dtype=int)
print("One-hot encoding complete.")

# Scale feature magnitudes to normalize data and speed up model convergence
print("Scaling numerical data...")
scaler = StandardScaler()

if use_kdd_test_file:
    # Fit scaler *only* on training data to prevent data leakage from the test dataset
    scaler.fit(combined_df.loc[:split_index-1, numerical_cols])
    combined_df[numerical_cols] = scaler.transform(combined_df[numerical_cols])
else:
    # Fit and transform the active single-dataset feature space
    combined_df[numerical_cols] = scaler.fit_transform(combined_df[numerical_cols])

print("Numerical data scaled.")

# ==========================================
# DATASET PARTITIONING
# ==========================================
if use_kdd_test_file:
    # Re-split combined data back into training and testing spaces based on original lengths
    X_train = combined_df.iloc[:split_index].drop(columns=['label'])
    y_train = combined_df.iloc[:split_index]['label']
    X_test = combined_df.iloc[split_index:].drop(columns=['label'])
    y_test = combined_df.iloc[split_index:]['label']
else:
    # Partition dataset into 80% training and 20% validation split with class stratification
    X = combined_df.drop(columns=['label'])
    y = combined_df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("KDDTrain+.txt split into 80% train / 20% test.")

print("Data preprocessed.")
print(f"Total features after encoding: {X_train.shape[1]}")
print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

# ==========================================
# MACHINE LEARNING MODEL TRAINING
# ==========================================
print("Training the Random Forest model...")
# Initialize Random Forest Classifier utilizing all available CPU cores (n_jobs=-1)
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, min_samples_leaf=1)

# Fit the classifier to training features and corresponding binary labels
model.fit(X_train, y_train)
print("Model training complete.")

# ==========================================
# SYSTEM EVALUATION & PERFORMANCE METRICS
# ==========================================
print("Evaluating the model on the test set...")
# Generate target predictions based on unseen validation/test feature inputs
y_pred = model.predict(X_test)

# Calculate and output overall accuracy percentage
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy * 100:.2f}%")

# Generate comprehensive classification report detailing Precision, Recall, and F1-Scores
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Attack (1)']))

# Construct and parse the Confusion Matrix for actionable cybersecurity insights
print("\nConfusion Matrix:")
print("         Predicted")
print("         Normal  Attack")
cm = confusion_matrix(y_test, y_pred)
print(f"Actual Normal: {cm[0]}")
print(f"Actual Attack: {cm[1]}")
print("------------------------")
print("This matrix shows:")
print(f"  {cm[0][0]} Normal connections were correctly classified as Normal (True Negatives)")
print(f"  {cm[0][1]} Normal connections were *incorrectly* classified as Attack (False Positives)")
print(f"  {cm[1][0]} Attack connections were *incorrectly* classified as Normal (False Negatives)")
print(f"  {cm[1][1]} Attack connections were correctly classified as Attack (True Positives)")