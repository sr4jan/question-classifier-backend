import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import os

# Set paths
data_path = 'data/professional_questions_core_extended_400.csv'
model_path = 'models/model.pkl'

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)

# Load and prepare data
print("Loading data...")
df = pd.read_csv(data_path)
print(f"Dataset shape: {df.shape}")

# Preprocess
df['Question'] = df['Question'].str.strip().str.lower()

# Split data
X = df['Question']
y = df['Category']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Create and train pipeline
print("Training model...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

# Verify model is fitted properly
print("Verifying model...")
try:
    # Test TF-IDF
    assert hasattr(pipeline.named_steps['tfidf'], 'vocabulary_')
    assert hasattr(pipeline.named_steps['tfidf'], 'idf_')
    print("✓ TF-IDF vectorizer is properly fitted")
    
    # Test classifier
    assert hasattr(pipeline.named_steps['clf'], 'coef_')
    print("✓ Classifier is properly fitted")
    
    # Test prediction
    test_text = X_test.iloc[0]
    pred = pipeline.predict([test_text])
    print(f"✓ Test prediction successful: {pred[0]}")
    
except AssertionError as e:
    print("ERROR: Model verification failed")
    raise e

# Evaluate
print("\nEvaluating model...")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
print(f"\nSaving model to {model_path}...")
joblib.dump(pipeline, model_path)

# Verify saved model
print("Verifying saved model...")
loaded_model = joblib.load(model_path)
try:
    test_text = X_test.iloc[0]
    pred = loaded_model.predict([test_text])
    print(f"✓ Saved model verification successful: {pred[0]}")
except Exception as e:
    print(f"ERROR: Saved model verification failed: {str(e)}")
    raise e

print("\nTraining completed successfully!")