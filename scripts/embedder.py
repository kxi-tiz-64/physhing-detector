import pandas as pd
import torch
import numpy as np
import os
import pickle
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# 1. Load mBERT
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')
model.eval()  # inference mode

# 2. Device configuration (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 3. Embed a single sentence
def get_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        # Use the CLS token embedding as sentence representation
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return cls_embedding

# 4. Load cleaned data
input_path = 'data/processed/cleaned.csv'
output_csv_path = 'data/processed/cleaned_with_embeddings.csv'
output_npy_path = 'data/processed/embeddings.npy'
output_pkl_path = 'data/processed/embeddings.pkl'

df = pd.read_csv(input_path)

# 5. Apply embedding to all rows
embeddings = []
print("🔄 Generating embeddings with mBERT...")
for text in tqdm(df['clean_text'], desc="Embedding"):
    emb = get_embedding(text)
    embeddings.append(emb)

# 6. Convert embeddings to DataFrame and concatenate with original data
embedding_df = pd.DataFrame(embeddings)

# 7. Save result in CSV
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
result_df = pd.concat([df.reset_index(drop=True), embedding_df.reset_index(drop=True)], axis=1)
result_df.to_csv(output_csv_path, index=False)
print(f"✅ Embeddings saved to {output_csv_path}")

# 8. Save embeddings as .npy (NumPy) and .pkl (Pickle)
np.save(output_npy_path, np.array(embeddings))
print(f"✅ Embeddings saved to {output_npy_path}")

with open(output_pkl_path, 'wb') as f:
    pickle.dump(embeddings, f)
print(f"✅ Embeddings saved to {output_pkl_path}")
