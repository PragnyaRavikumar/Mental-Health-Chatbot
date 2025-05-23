import pandas as pd

# Load the dataset
df = pd.read_csv("data/goemotions_1.csv")

# Define emotion columns (30 emotions)
emotion_columns = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
    'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love',
    'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse',
    'sadness', 'surprise', 'neutral'
]

# Keep only necessary columns
df = df[['text'] + emotion_columns]

# Remove rows that have no emotion at all
df = df[df[emotion_columns].sum(axis=1) > 0]

# Create a new column with list of emotions (multi-labels)
df['labels'] = df[emotion_columns].apply(lambda row: [emotion for emotion in emotion_columns if row[emotion] == 1], axis=1)

# Save the final dataset for training
df[['text', 'labels']].to_json("data/multilabel_data.json", orient="records", lines=True)

print("✅ Saved multi-label training data as JSON!")
