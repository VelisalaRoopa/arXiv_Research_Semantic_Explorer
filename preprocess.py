import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = str(text).lower()

    tokens = word_tokenize(text)

    tokens = [
        word
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]

    return tokens