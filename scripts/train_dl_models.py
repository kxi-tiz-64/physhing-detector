import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# Set seed for reproducibility
torch.manual_seed(42)

# Define the LSTM, CNN, and MLP models
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

# Define the function to train the models
def train(model, optimizer, criterion, dataloader):
    model.train()
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

# Define the function to evaluate the models
def evaluate(model, dataloader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.numpy())
            targets.extend(y_batch.numpy())
    acc = accuracy_score(targets, preds)
    report = classification_report(targets, preds)
    cm = confusion_matrix(targets, preds)
    return acc, report, cm

# Add this part to ensure the training script only runs when executed directly
if __name__ == "__main__":

    print("📄 Loading dataset...")
    df = pd.read_csv("data/processed/embedded_emails.csv")

    # Clean any NaNs
    print("🧹 Cleaning data...")
    df = df.dropna(subset=['label'])
    X = df[[f'embedding_{i}' for i in range(768)]].values
    y = df['label'].values

    # Split
    print("✂️ Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    test_ds = TensorDataset(X_test_tensor, y_test_tensor)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=64)

    # Train and evaluate LSTM
    print("🚀 Training LSTM...")
    lstm_model = LSTMModel()
    optimizer = optim.Adam(lstm_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        train(lstm_model, optimizer, criterion, train_dl)
        acc, _, _ = evaluate(lstm_model, test_dl)
        print(f"📈 LSTM Epoch {epoch+1}: Accuracy = {acc:.4f}")

    acc, report, cm = evaluate(lstm_model, test_dl)
    print("\n📊 LSTM Final Report:\n", report)
    print("🧩 Confusion Matrix:\n", cm)

    # Train and evaluate fine-tuned CNN
    print("\n🚀 Training CNN...")
    cnn_model = CNNModel()
    optimizer = optim.Adam(cnn_model.parameters(), lr=1e-3)

    for epoch in range(5):
        train(cnn_model, optimizer, criterion, train_dl)
        acc, _, _ = evaluate(cnn_model, test_dl)
        print(f"📈 CNN Epoch {epoch+1}: Accuracy = {acc:.4f}")

    acc, report, cm = evaluate(cnn_model, test_dl)
    print("\n📊 CNN Final Report:\n", report)
    print("🧩 Confusion Matrix:\n", cm)

    # Train and evaluate MLP
    print("\n🚀 Training MLP...")
    mlp_model = MLPModel()
    optimizer = optim.Adam(mlp_model.parameters(), lr=1e-3)

    for epoch in range(5):
        train(mlp_model, optimizer, criterion, train_dl)
        acc, _, _ = evaluate(mlp_model, test_dl)
        print(f"📈 MLP Epoch {epoch+1}: Accuracy = {acc:.4f}")

    acc, report, cm = evaluate(mlp_model, test_dl)
    print("\n📊 MLP Final Report:\n", report)
    print("🧩 Confusion Matrix:\n", cm)

    # Save models
    os.makedirs("models", exist_ok=True)
    torch.save(lstm_model.state_dict(), "models/lstm_model.pth")
    torch.save(cnn_model.state_dict(), "models/cnn_model.pth")
    torch.save(mlp_model.state_dict(), "models/mlp_model.pth")
    print("💾 Models saved to 'models/' directory!")
