import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# ======================
# Load model and vectorizer
# ======================
model = joblib.load("models/logreg_model.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

# ======================
# Load saved test set
# ======================
X_test = pd.read_csv("data/X_test.csv")["text"].astype(str)
y_test = pd.read_csv("data/y_test.csv")["label"]

# ======================
# Vectorization
# ======================
X_test_tfidf = vectorizer.transform(X_test)

# ======================
# Prediction
# ======================
y_pred = model.predict(X_test_tfidf)

# ======================
# Evaluation
# ======================
print("✅ Accuracy on Test Set:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
