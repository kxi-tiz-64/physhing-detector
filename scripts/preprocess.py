import pandas as pd
import re
import os

# Step 1: Define a cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)         # remove URLs
    text = re.sub(r'<.*?>', '', text)                         # remove HTML tags
    text = re.sub(r'\W+', ' ', text)                          # remove non-word characters
    text = re.sub(r'\s+', ' ', text)                          # remove extra spaces
    return text.strip()

# Step 2: Load and clean the dataset
def load_and_clean(filepath):
    df = pd.read_csv(filepath)

    # Check if 'text' column exists
    if 'text' not in df.columns:
        raise KeyError("Column 'text' not found in the dataset. Please check your CSV headers.")

    # Drop rows with missing text
    df = df.dropna(subset=['text'])

    # Apply text cleaning
    df['clean_text'] = df['text'].apply(clean_text)

    return df

# Step 3: Save the cleaned data
if __name__ == '__main__':
    input_path = 'data/raw/vectorized.csv'
    output_path = 'data/processed/cleaned.csv'

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Clean the data and save it
    df_clean = load_and_clean(input_path)
    df_clean.to_csv(output_path, index=False)

    print(f'✅ Cleaned data saved to {output_path}')
