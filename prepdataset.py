import pandas as pd
import os

# Load your dataset
df = pd.read_csv(r"D:\genai\data\WELFake_Dataset.csv")

# Drop 'Unnamed: 0' if it exists
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Keep only text + label (if title exists, you can drop it too)
df = df[['text', 'label']]

# Make sure the 'data' folder exists
os.makedirs("data", exist_ok=True)

# Save cleaned dataset
df.to_csv("data/news.csv", index=False)

print("✅ Saved cleaned dataset to data/news.csv")
