import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.sparse import hstack

# Load data
df = pd.read_csv("../data/train.csv")

df['journal_text'] = df['journal_text'].fillna("")

# Features
tfidf = TfidfVectorizer(max_features=3000)
X_text = tfidf.fit_transform(df['journal_text'])

meta_cols = ['sleep_hours', 'energy_level', 'stress_level', 'duration_min']
X_meta = df[meta_cols].fillna(0)

X = hstack([X_text, X_meta])

# Targets
y_state = df['emotional_state']
y_intensity = df['intensity']

# Models
model_state = RandomForestClassifier()
model_intensity = RandomForestRegressor()

model_state.fit(X, y_state)
model_intensity.fit(X, y_intensity)

# Save
joblib.dump(tfidf, "../models/tfidf.pkl")
joblib.dump(model_state, "../models/state_model.pkl")
joblib.dump(model_intensity, "../models/intensity_model.pkl")

print("Training complete ✅")
