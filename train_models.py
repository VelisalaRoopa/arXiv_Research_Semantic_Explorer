import pandas as pd
from gensim.models import Word2Vec
from preprocess import preprocess_text

# Load dataset
df = pd.read_csv("arXiv_scientific_dataset.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)

# Remove rows with missing summaries
df = df.dropna(subset=["summary"])

# Preprocess text
df["tokens"] = df["summary"].astype(str).apply(preprocess_text)

# Convert tokenized summaries into a list of sentences
sentences = df["tokens"].tolist()

print("Training Word2Vec model...")

# Train Word2Vec
model = Word2Vec(
    sentences=sentences,
    vector_size=100,   # Embedding dimension
    window=5,          # Context window
    min_count=2,       # Ignore words with frequency < 2
    workers=4,         # CPU cores
    sg=0               # 0 = CBOW, 1 = Skip-Gram
)

# Save model
model.save("word2vec.model")

print("✅ Word2Vec model saved as 'word2vec.model'")
print("Vocabulary Size:", len(model.wv.index_to_key))