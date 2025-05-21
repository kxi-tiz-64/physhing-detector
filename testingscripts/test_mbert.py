from transformers import AutoTokenizer, AutoModel

model_name = "bert-base-multilingual-cased"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    print("✅ mBERT loaded successfully!")
except Exception as e:
    print("❌ Error loading mBERT:", e)
