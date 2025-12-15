
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set up output directory for EDA
OUTPUT_DIR = 'eda_outputs'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def perform_eda(df):
    print("\n--- Exploratory Data Analysis ---")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe(include='all'))
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Visualizing Target Variable
    if 'approval_status' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x='approval_status', data=df)
        plt.title('Distribution of Approval Status')
        plt.savefig(os.path.join(OUTPUT_DIR, 'approval_status_dist.png'))
        plt.close()
    
    # Correlation Heatmap (numerical features)
    numerical_df = df.select_dtypes(include=[np.number])
    if not numerical_df.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap')
        plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'))
        plt.close()

    print(f"EDA plots saved to {OUTPUT_DIR}")

def preprocessing(df):
    print("\n--- Preprocessing ---")
    
    # Drop Loan_ID as it's not useful for prediction
    if 'application_id' in df.columns:
        df = df.drop('application_id', axis=1)
        print("Dropped 'application_id'.")
    
    # Explicitly select only expected columns to avoid "Unnamed" empty columns
    expected_cols = [
        'applicant_gender', 'is_married', 'num_dependents', 'education_level', 
        'is_self_employed', 'primary_income', 'secondary_income', 
        'loan_amount_requested', 'term_duration_months', 'has_credit_history', 
        'residence_area', 'approval_status'
    ]
    # Filter only columns that exist in the dataframe (for robustness)
    cols_to_keep = [col for col in expected_cols if col in df.columns]
    df = df[cols_to_keep]
    print(f"Dataframe filtered. Current shape: {df.shape}")
    
    # Data Cleaning: Convert '3+' in dependants to '3' if necessary, but LabelEncoder handles strings fine.
    # However, strict numerical conversion might be better for models. 
    # For this pass, we treat it as categorical.

    # Handling Missing Values
    # 1. Numerical: Fill with Median
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Filled numerical '{col}' NaNs with median: {median_val}")

    # 2. Categorical: Fill with Mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
                print(f"Filled categorical '{col}' NaNs with mode: {mode_val[0]}")
            else:
                df[col] = df[col].fillna("Unknown")
                print(f"Filled categorical '{col}' NaNs with 'Unknown' (no mode found)")
        
        # Force to string to avoid TypeError in LabelEncoder with mixed types
        df[col] = df[col].astype(str)
    
    # Final check for NaNs
    if df.isnull().sum().sum() > 0:
        print("WARNING: NaNs still present after filling!")
        print(df.isnull().sum()[df.isnull().sum() > 0])
        # Fallback fill
        df = df.fillna(0) 

    # Encoding Categorical Variables
    le_dict = {}
    le = LabelEncoder()
    encoded_cols = []
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
        encoded_cols.append(col)
        le_dict[col] = le
        if col == 'approval_status':
            print(f"Target '{col}' encoded mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
            
    print(f"Encoded columns: {encoded_cols}")
    return df, le_dict

def train_model(df):
    print("\n--- Model Training ---")
    
    if 'approval_status' not in df.columns:
        print("Error: Target variable 'approval_status' not found!")
        return None, None, None, None, None

    X = df.drop('approval_status', axis=1)
    y = df['approval_status']
    
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    # Check for NaNs in X
    if X.isnull().values.any():
        print("Error: Input data contains NaNs.")
        print(X.isnull().sum())
        return None, None, None, None, None

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model: Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    print("Random Forest model trained.")
    
    # Model: Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    print("Logistic Regression model trained.")
    
    return rf_model, lr_model, X_test_scaled, y_test, scaler

def evaluate_model(model, X_test, y_test, model_name="Model"):
    if model is None:
        print(f"Skipping evaluation for {model_name} (Model is None).")
        return

    print(f"\n--- Evaluating {model_name} ---")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Save Confusion Matrix plot
    try:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(OUTPUT_DIR, f'confusion_matrix_{model_name.replace(" ", "_")}.png'))
        plt.close()
    except Exception as e:
        print(f"Could not save confusion matrix plot: {e}")

import pickle

def save_artifacts(model, scaler, le_dict):
    print("\n--- Saving Artifacts ---")
    try:
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print("Model saved to model.pkl")
        
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print("Scaler saved to scaler.pkl")
        
        with open('encoders.pkl', 'wb') as f:
            pickle.dump(le_dict, f)
        print("Label Encoders saved to encoders.pkl")
    except Exception as e:
        print(f"Error saving artifacts: {e}")

if __name__ == "__main__":
    file_path = 'financial_lending.csv'
    
    if os.path.exists(file_path):
        df = load_data(file_path)
        
        try:
            perform_eda(df)
        except Exception as e:
            print(f"EDA failed: {e}")
        
        try:
            df_processed, le_dict = preprocessing(df)
            
            rf, lr, X_test, y_test, scaler = train_model(df_processed)
            
            if rf is not None:
                evaluate_model(rf, X_test, y_test, "Random Forest")
                
                # Feature Importance
                importances = rf.feature_importances_
                feature_names = df_processed.drop('approval_status', axis=1).columns
                indices = np.argsort(importances)[::-1]
                
                plt.figure(figsize=(10, 6))
                plt.title("Feature Importances (Random Forest)")
                plt.bar(range(X_test.shape[1]), importances[indices], align="center")
                plt.xticks(range(X_test.shape[1]), feature_names[indices], rotation=90)
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'))
                plt.close()
                print("Feature importance plot saved.")
                
                # Save Artifacts
                save_artifacts(rf, scaler, le_dict)
                
            if lr is not None:
                evaluate_model(lr, X_test, y_test, "Logistic Regression")

        except Exception as e:
            print(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()

    else:
        print(f"Error: File {file_path} not found.")
