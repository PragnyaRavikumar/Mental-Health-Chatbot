import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Load dataset
df = pd.read_csv("data/goemotions_merged.csv")

# Extract text and labels
texts = df['text'].tolist()
label_cols = df.columns[9:]  # Assuming labels start from column 10 onward
labels = df[label_cols].values.tolist()

# Binarize labels
mlb = MultiLabelBinarizer()
multi_labels = [tuple(label_cols[i] for i, v in enumerate(row) if v == 1) for row in labels]
binary_labels = mlb.fit_transform(multi_labels)

# Save the binarizer
import joblib
joblib.dump(mlb, "model/multilabel_binarizer_bert.pkl")

# Tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

dataset = EmotionDataset(texts, binary_labels, tokenizer)

# Model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(label_cols),
    problem_type="multi_label_classification"
)

# Training args
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=500,
    save_total_limit=2
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train model
trainer.train()

# Save model and tokenizer
model.save_pretrained("model/bert_emotion_model")
tokenizer.save_pretrained("model/bert_emotion_model")
