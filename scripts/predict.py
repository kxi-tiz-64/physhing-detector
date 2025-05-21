# predict2.py (final, fully fixed and integrated for web app)
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
import re

MODEL_NAME = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
embedding_model = AutoModel.from_pretrained(MODEL_NAME)


# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model.to(device)
embedding_model.eval()

# Define deep models (exact same structure as training)
class MLPModel(nn.Module):
    def __init__(self, input_size=768, num_classes=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class LSTMModel(nn.Module):
    def __init__(self, input_size=768, hidden_size=128, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out

class CNNModel(nn.Module):
    def __init__(self, input_size=768, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(2)
        x = self.dropout(x)
        return self.fc(x)

# Load trained models
mlp = MLPModel().to(device)
cnn = CNNModel().to(device)
lstm = LSTMModel().to(device)

mlp.load_state_dict(torch.load("models/mlp_model.pth", map_location=device))
cnn.load_state_dict(torch.load("models/cnn_model.pth", map_location=device))
lstm.load_state_dict(torch.load("models/lstm_model.pth", map_location=device))

mlp.eval()
cnn.eval()
lstm.eval()

# Clean and preprocess text
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

# Generate mBERT/XLM-RoBERTa embeddings
def get_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        output = embedding_model(**tokens)
    cls_embedding = output.last_hidden_state[:, 0, :]  # CLS token
    return cls_embedding.squeeze(0).cpu().numpy()

# Predict using ensemble
LABEL_MAP = {0: "Legitimate", 1: "Phishing"}

def predict_email(text):
    text = clean_text(text)
    embedding = get_embedding(text)
    input_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = [
            mlp(input_tensor),
            cnn(input_tensor),
            lstm(input_tensor)
        ]
        probs = [torch.softmax(o, dim=1) for o in outputs]
        avg_prob = torch.mean(torch.stack(probs), dim=0)
        pred_label = torch.argmax(avg_prob, dim=1).item()

    return LABEL_MAP[pred_label], avg_prob.squeeze().cpu().numpy().tolist()

# Hard voting prediction
def predict_email_hard_vote(text):
    text = clean_text(text)
    embedding = get_embedding(text)
    input_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = [
            mlp(input_tensor),
            cnn(input_tensor),
            lstm(input_tensor)
        ]
        predictions = [torch.argmax(output, dim=1).item() for output in outputs]
        final_label = max(set(predictions), key=predictions.count)

    return LABEL_MAP[final_label], predictions