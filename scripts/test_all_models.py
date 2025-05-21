from huggingface_hub import load_torch_model
import pandas as pd
import torch
import pickle
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score

from train_dl_models import LSTMModel, MLPModel

# Load the dataset
print("📄 Loading dataset...")
df = pd.read_csv('data/processed/embedded_multilingual_test.csv')  # Adjust path as necessary

# Data cleaning
print("🧹 Cleaning data...")
df = df.dropna(subset=['clean_text'])  # Ensure no NaN in clean_text column
df['labels'] = df['labels'].apply(lambda x: int(x))  # Ensure labels are integers

# Prepare the embeddings and labels
X_test = df[[f'embedding_{i}' for i in range(768)]].values  # Use the 768 embeddings directly
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test = df['labels'].tolist()

# Initialize model dictionary
models = {}

# Load LSTM model
print("🚀 Loading LSTM model...")
lstm_model = LSTMModel()
lstm_model.load_state_dict(torch.load('models/lstm_model.pth'))  # Replace with actual path to model
lstm_model.eval()  # Set the model to evaluation mode
models['lstm'] = lstm_model

print("🚀 Loading MLP model...")
mlp_model = MLPModel()
mlp_model.load_state_dict(torch.load('models/mlp_model.pth'))  # Replace with actual path to model
mlp_model.eval()  # Set the model to evaluation mode
models['mlp'] = mlp_model

# Predict with each model
print("🔮 Making predictions with each model...")
all_preds = []
for model_name, model in models.items():
    print(f"Testing {model_name} model...")
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, preds = torch.max(outputs, 1)  # Get predicted class index
        all_preds.append(preds)

# Combine predictions (Majority Voting or Simple Ensemble)
final_preds = torch.mode(torch.stack(all_preds), dim=0)[0]  # Majority voting

# Evaluate performance
print("📊 Evaluation results:")
# Since this is a phishing dataset, we care about detecting phishing emails (label 1)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, final_preds.numpy(), average='binary', pos_label=1)

print(f"Precision (for phishing): {precision:.4f}")
print(f"Recall (for phishing): {recall:.4f}")
print(f"F1 Score (for phishing): {f1:.4f}")
print(f"Accuracy: {accuracy_score(y_test, final_preds.numpy()):.4f}")

# Save results (optional)
with open('predictions.pkl', 'wb') as f:
    pickle.dump(final_preds.numpy(), f)

print("✅ Testing complete.")
