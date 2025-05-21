import os
import logging
import glob
import re
import zipfile
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    filename='embedder.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Load mBERT
print("🧠 Loading mBERT model...")
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"🚀 Using device: {device}")
logging.info(f"Using device: {device}")

# Load data
print("📄 Loading cleaned dataset...")
df_full = pd.read_csv('data/processed/cleaned.csv').dropna(subset=['clean_text']).reset_index(drop=True)

# Resume from checkpoint
os.makedirs("data/multilingual", exist_ok=True)
os.makedirs("data/scripts", exist_ok=True)

checkpoint_files = glob.glob("data/multilingual/embedded_checkpoint_*.csv")
resume_from_idx = 0
embeddings = []

if checkpoint_files:
    latest_ckpt = max(checkpoint_files, key=lambda x: int(re.findall(r'\d+', x)[-1]))
    resume_from_idx = int(re.findall(r'\d+', latest_ckpt)[-1])
    print(f"🔁 Resuming from checkpoint: {latest_ckpt} (row {resume_from_idx})")
    logging.info(f"Resuming from checkpoint: {latest_ckpt}")
    checkpoint_df = pd.read_csv(latest_ckpt)
    df_full = df_full.iloc[resume_from_idx:].reset_index(drop=True)
    embeddings = checkpoint_df[[f'embedding_{i}' for i in range(768)]].values.tolist()
else:
    print("🆕 No checkpoint found. Starting from scratch.")
    logging.info("Starting fresh.")

texts = df_full['clean_text'].tolist()
intermediate_save_interval = 2000
batch_size = 16

def batch_embed(batch_texts):
    try:
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return [emb for emb in batch_embeddings]
    except Exception as e:
        logging.error(f"❌ Batch embedding error: {e}")
        return [np.zeros(768) for _ in batch_texts]

print("🔄 Generating embeddings...")
for start_idx in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
    end_idx = start_idx + batch_size
    global_idx = resume_from_idx + start_idx + 1

    batch_texts = texts[start_idx:end_idx]
    batch_embs = batch_embed(batch_texts)

    # Validate and append
    for emb in batch_embs:
        emb = np.array(emb)
        if emb.shape == (768,):
            embeddings.append(emb)
        else:
            embeddings.append(np.zeros(768))

    # Save intermediate
    if global_idx % intermediate_save_interval == 0 or end_idx >= len(texts):
        temp_df = pd.read_csv('data/processed/cleaned.csv').iloc[:resume_from_idx + end_idx].reset_index(drop=True)
        try:
            stacked = np.vstack(embeddings)
            emb_df = pd.DataFrame(stacked, columns=[f'embedding_{j}' for j in range(768)])
            combined = pd.concat([temp_df, emb_df], axis=1)
            ckpt_path = f"data/multilingual/embedded_checkpoint_{resume_from_idx + end_idx}.csv"
            combined.to_csv(ckpt_path, index=False)
            logging.info(f"💾 Saved checkpoint: {ckpt_path}")
        except Exception as e:
            logging.error(f"⚠️ Failed to save checkpoint at {global_idx}: {e}")

# Final save
print("💾 Saving final result...")
try:
    full_df = pd.read_csv('data/processed/cleaned.csv').reset_index(drop=True)
    stacked = np.vstack(embeddings)
    emb_df = pd.DataFrame(stacked, columns=[f'embedding_{j}' for j in range(768)])
    final_df = pd.concat([full_df, emb_df], axis=1)

    final_csv_path = "data/processed/embedded_emails.csv"
    final_df.to_csv(final_csv_path, index=False)

    # Zip it
    zip_path = "data/processed/embedded_emails.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(final_csv_path, arcname=os.path.basename(final_csv_path))


    print("✅ Final file saved & zipped at:")
    print("   - CSV: data/scripts/embedded_emails.csv")
    print("   - ZIP: data/scripts/embedded_emails.zip")
    logging.info("✅ Final save completed.")

    # Cleanup intermediate checkpoints
    for f in checkpoint_files + glob.glob("data/multilingual/embedded_checkpoint_*.csv"):
        os.remove(f)
    print("🧹 Deleted intermediate checkpoint files.")

except Exception as e:
    print(f"❌ Final save failed: {e}")
    logging.error(f"❌ Final save failed: {e}")
