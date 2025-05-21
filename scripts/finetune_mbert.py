# scripts/fine_tune_mbert.py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Load data
df = pd.read_csv("data/processed/cleaned.csv")
df = df.dropna(subset=["clean_text", "label"])

X_train, X_val, y_train, y_val = train_test_split(df["clean_text"], df["label"], test_size=0.2, stratify=df["label"])

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
train_ds = EmailDataset(X_train.tolist(), y_train.tolist(), tokenizer)
val_ds = EmailDataset(X_val.tolist(), y_val.tolist(), tokenizer)

train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=16)

model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
EPOCHS = 3
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in train_dl:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"🔥 Epoch {epoch + 1}: Loss = {total_loss / len(train_dl):.4f}")

# Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_dl:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\n📊 Classification Report:\n", classification_report(all_labels, all_preds))

# Save model
model.save_pretrained("models/fine_tuned_mbert")
tokenizer.save_pretrained("models/fine_tuned_mbert")
print("✅ Fine-tuned model saved to models/fine_tuned_mbert")
