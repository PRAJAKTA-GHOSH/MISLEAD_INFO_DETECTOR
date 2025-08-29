import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ======================
# Load Dataset
# ======================
data = pd.read_csv("D:/genai/data/WELFake_Dataset.csv")
print("Dataset loaded successfully!")

# ======================
# Preprocessing
# ======================
# Drop rows with missing 'text' or 'label'
data = data.dropna(subset=['text', 'label'])

# Convert text to string, lowercase, and strip spaces
data['text'] = data['text'].astype(str).str.lower().str.strip()

# Remove empty strings
data = data[data['text'] != '']

# Features and labels
X = data['text']
y = data['label']

# ======================
# Train-test split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save test set for consistent evaluation
X_test.to_csv("data/X_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

# ======================
# Vectorization
# ======================
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ======================
# Model Training
# ======================
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# ======================
# Evaluation
# ======================
y_pred = model.predict(X_test_tfidf)
print("✅ Training Accuracy on Test Set:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# ======================
# Save Model and Vectorizer
# ======================
joblib.dump(model, "models/logreg_model.joblib")
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
print("✅ Model and vectorizer saved successfully!")
