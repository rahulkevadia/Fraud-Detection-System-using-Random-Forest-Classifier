import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import joblib
import logging
import sys
import gc  # Garbage collector for memory management
from flask import Flask, render_template, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

class SafeLabelEncoder(LabelEncoder):
    def fit(self, y):
        return super().fit(np.append(y, ['unknown']))
    
    def transform(self, y):
        y = np.array(y)
        mask = np.isin(y, self.classes_)
        y_copy = y.copy()
        y_copy[~mask] = 'unknown'
        return super().transform(y_copy)

# Load the trained model and preprocessing objects
def load_model():
    try:
        model = joblib.load('fraud_detection_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        return model, scaler, label_encoders
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return None, None, None

@app.route('/')
def home():
    return render_template('home.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load model and preprocessing objects
        model, scaler, label_encoders = load_model()
        if model is None:
            return render_template('home.html', error="Model could not be loaded")

        # Get form data
        data = {
            'type': request.form['type'],
            'amount': float(request.form['amount']),
            'nameOrig': request.form['nameOrig'],
            'oldbalanceOrg': float(request.form['oldbalanceOrg']),
            'newbalanceOrig': float(request.form['newbalanceOrig']),
            'nameDest': request.form['nameDest'],
            'oldbalanceDest': float(request.form['oldbalanceDest']),
            'newbalanceDest': float(request.form['newbalanceDest'])
        }

        # Create DataFrame
        df = pd.DataFrame([data])

        # Apply label encoding
        for col in ['type', 'nameOrig', 'nameDest']:
            if col in label_encoders:
                df[col] = label_encoders[col].transform(df[col].astype(str))

        # Scale features
        df_scaled = scaler.transform(df)

        # Make prediction
        prediction = model.predict(df_scaled)[0]
        probability = model.predict_proba(df_scaled)[0][1]  # Probability of fraud

        return render_template('home.html', 
                             prediction=prediction,
                             probability=probability,
                             form_data=data)

    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return render_template('home.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)

# Original model training code...
try:
    if not joblib.load('fraud_detection_model.pkl'):
        # Load the dataset
        logging.info("Loading dataset...")
        
        # Calculate total rows and determine sample size
        total_rows = sum(1 for _ in open('Dataset/Transactions Dataset.csv')) - 1
        logging.info(f"Total rows in dataset: {total_rows}")
        
        # Use smaller sample size for faster processing
        sample_size = min(50000, int(total_rows * 0.02))  # Reduced to 2% or 50k rows max
        skip_rows = sorted(np.random.choice(range(1, total_rows + 1), 
                                          total_rows - sample_size, 
                                          replace=False))
        
        logging.info(f"Using a sample size of {sample_size} rows for model training")
        
        # Read the sampled data directly
        df = pd.read_csv('Dataset/Transactions Dataset.csv',
                         skiprows=skip_rows)
        
        # Identify columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        
        logging.info(f"Categorical columns: {categorical_columns.tolist()}")
        
        # Handle missing values in numeric columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        # Initialize and fit label encoders
        label_encoders = {}
        for col in categorical_columns:
            if df[col].dtype == 'object':
                label_encoders[col] = SafeLabelEncoder()
                df[col] = label_encoders[col].fit_transform(df[col].astype(str))
        
        logging.info(f"Dataset processed successfully with {len(df)} rows")
        
        # Separate features and target variable
        if 'isFraud' not in df.columns:
            raise ValueError("Target variable 'isFraud' not found in the dataset")
        
        X = df.drop('isFraud', axis=1)
        y = df['isFraud']
        
        # Free up memory
        del df
        gc.collect()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Free up memory
        del X, y
        gc.collect()
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Free up memory
        del X_train
        gc.collect()
        
        # Apply SMOTE with reduced sample size
        logging.info("Applying SMOTE for handling imbalanced data...")
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
        
        # Free up memory
        del X_train_scaled, y_train
        gc.collect()
        
        # Train lighter Random Forest model
        logging.info("Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=50,  # Reduced number of trees
            max_depth=8,      # Reduced max depth
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,        # Use all CPU cores
            class_weight='balanced'  # Handle class imbalance
        )
        
        model.fit(X_train_res, y_train_res)
        logging.info("Model training completed successfully!")
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Model evaluation
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        
        logging.info("\nModel Evaluation:")
        logging.info(f'Accuracy: {accuracy:.4f}')
        logging.info(f'ROC-AUC Score: {roc_auc:.4f}')
        logging.info('\nConfusion Matrix:')
        logging.info(f'\n{confusion_matrix(y_test, y_pred)}')
        logging.info('\nClassification Report:')
        logging.info(f'\n{classification_report(y_test, y_pred)}')
        
        # Feature importance visualization
        feature_importances = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importances['Feature'][:10], feature_importances['Importance'][:10])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Top 10 Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
        # Save the model and preprocessing objects
        logging.info("Saving model and preprocessing objects...")
        try:
            joblib.dump(model, 'fraud_detection_model.pkl')
            logging.info("Model saved successfully as 'fraud_detection_model.pkl'")
            
            joblib.dump(scaler, 'scaler.pkl')
            logging.info("Scaler saved successfully as 'scaler.pkl'")
            
            joblib.dump(label_encoders, 'label_encoders.pkl')
            logging.info("Label encoders saved successfully as 'label_encoders.pkl'")
            
        except Exception as e:
            logging.error(f"Error saving model files: {str(e)}")
            raise
        
        logging.info("Model training and evaluation completed successfully!")

except Exception as e:
    logging.error(f"An error occurred: {str(e)}")
    sys.exit(1)

# Function to make predictions on new data
def predict_fraud(new_data_path):
    try:
        # Load the model and preprocessing objects
        model = joblib.load('fraud_detection_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        
        # Process new data in chunks
        chunk_size = 10000
        predictions_list = []
        probabilities_list = []
        
        for chunk in pd.read_csv(new_data_path, chunksize=chunk_size):
            # Apply label encoding
            for col in label_encoders.keys():
                if col in chunk.columns:
                    chunk[col] = label_encoders[col].transform(chunk[col].astype(str))
            
            # Scale the features
            chunk_scaled = scaler.transform(chunk)
            
            # Make predictions
            chunk_predictions = model.predict(chunk_scaled)
            chunk_probabilities = model.predict_proba(chunk_scaled)[:, 1]
            
            predictions_list.extend(chunk_predictions)
            probabilities_list.extend(chunk_probabilities)
        
        return np.array(predictions_list), np.array(probabilities_list)
        
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return None, None
