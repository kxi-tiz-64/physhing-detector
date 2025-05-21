from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import joblib
import numpy as np
import re

# Flask setup
app = Flask(__name__)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and embedding model
MODEL_NAME = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
embedding_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
embedding_model.eval()

# Preprocessing
def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

def get_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        output = embedding_model(**tokens)
    cls_embedding = output.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze(0).cpu().numpy()

# Label map
LABEL_MAP = {0: "Legitimate", 1: "Phishing"}

# Define DL model classes
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

class LSTMModel(nn.Module):
    def __init__(self, input_size=768, hidden_size=128, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# Load DL models
mlp = MLPModel().to(device)
cnn = CNNModel().to(device)
lstm = LSTMModel().to(device)

mlp.load_state_dict(torch.load("models/mlp_model.pth", map_location=device))
cnn.load_state_dict(torch.load("models/cnn_model.pth", map_location=device))
lstm.load_state_dict(torch.load("models/lstm_model.pth", map_location=device))

mlp.eval()
cnn.eval()
lstm.eval()

# Load ML models
rf = joblib.load("models/random_forest_model.pkl")
svm = joblib.load("models/svm_model.pkl")
lr = joblib.load("models/log_reg_model.pkl")

# Prediction logic
def hybrid_predict(text):
    text = clean_text(text)
    embedding = get_embedding(text)
    tensor_input = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
    embedding_np = embedding.reshape(1, -1)

    with torch.no_grad():
        dl_preds = [
            torch.argmax(model(tensor_input), dim=1).item()
            for model in [mlp, cnn, lstm]
        ]
    ml_preds = [
        rf.predict(embedding_np)[0],
        svm.predict(embedding_np)[0],
        lr.predict(embedding_np)[0]
    ]
    final_votes = dl_preds + ml_preds
    prediction = max(set(final_votes), key=final_votes.count)

    # Return a string instead of an index
    return LABEL_MAP[prediction]  # "Phishing" or "Legitimate"


# Routes
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    email = data.get("email", "")
    if not email.strip():
        return jsonify({"error": "No email content provided."}), 400
    prediction = hybrid_predict(email)
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)