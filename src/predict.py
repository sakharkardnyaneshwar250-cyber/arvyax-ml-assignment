import pandas as pd
import joblib
from scipy.sparse import hstack
from decision import decide_action

# Load
df = pd.read_csv("../data/test.csv")

tfidf = joblib.load("../models/tfidf.pkl")
model_state = joblib.load("../models/state_model.pkl")
model_intensity = joblib.load("../models/intensity_model.pkl")

df['journal_text'] = df['journal_text'].fillna("")

# Features
X_text = tfidf.transform(df['journal_text'])
meta_cols = ['sleep_hours', 'energy_level', 'stress_level', 'duration_min']
X_meta = df[meta_cols].fillna(0)

X = hstack([X_text, X_meta])

# Predictions
state_pred = model_state.predict(X)
intensity_pred = model_intensity.predict(X)

probs = model_state.predict_proba(X)
confidence = probs.max(axis=1)
uncertain_flag = (confidence < 0.6).astype(int)

# Decision layer
actions = []
times = []

for i in range(len(df)):
    action, when = decide_action(
        state_pred[i],
        intensity_pred[i],
        df['stress_level'].iloc[i],
        df['energy_level'].iloc[i],
        df['time_of_day'].iloc[i]
    )
    actions.append(action)
    times.append(when)

# Output
output = pd.DataFrame({
    "id": df["id"],
    "predicted_state": state_pred,
    "predicted_intensity": intensity_pred,
    "confidence": confidence,
    "uncertain_flag": uncertain_flag,
    "what_to_do": actions,
    "when_to_do": times
})

output.to_csv("../predictions.csv", index=False)

print("Prediction complete ✅")
