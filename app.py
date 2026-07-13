import streamlit as st
import pandas as pd

from utils import (
    get_similar_words,
    get_vector,
    similarity
)

# -----------------------------------------------------
# Page Configuration
# -----------------------------------------------------
st.set_page_config(
    page_title="arXiv Research Word2Vec Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------
# Custom CSS
# -----------------------------------------------------
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.main-header{
    background: linear-gradient(90deg,#0F172A,#1D4ED8);
    padding:30px;
    border-radius:15px;
    color:white;
    text-align:center;
    margin-bottom:20px;
}

.main-header h1{
    font-size:42px;
    margin-bottom:5px;
}

.main-header p{
    font-size:18px;
    color:#E5E7EB;
}

.metric-card{
    background:#F8FAFC;
    border-radius:12px;
    padding:18px;
    border:1px solid #E5E7EB;
    text-align:center;
}

.result-box{
    background:#F1F5F9;
    border-radius:10px;
    padding:15px;
}

footer{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# Header
# -----------------------------------------------------
st.markdown("""
<div class="main-header">

<h1>🧠 arXiv Research Word2Vec Explorer</h1>

<p>
Explore semantic relationships between scientific terms using
Word2Vec embeddings trained on arXiv research paper abstracts.
</p>

</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# Dashboard Cards
# -----------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("📚 Dataset", "arXiv")

with c2:
    st.metric("🤖 Model", "Word2Vec")

with c3:
    st.metric("📄 Corpus", "Research Abstracts")

with c4:
    st.metric("⚡ Framework", "Streamlit")

st.divider()

# -----------------------------------------------------
# Sidebar
# -----------------------------------------------------

st.sidebar.title("🧭 Navigation")

menu = st.sidebar.radio(
    "Choose Feature",
    [
        "🔍 Similar Words",
        "🧠 Word Vector",
        "📈 Word Similarity"
    ]
)

st.sidebar.markdown("---")

st.sidebar.subheader("📌 About Project")

st.sidebar.write("""
This application explores semantic relationships between
scientific terms using a Word2Vec model trained on
arXiv research paper abstracts.

### Technologies

- Python
- Gensim
- Streamlit
- NLTK

### Try Searching

- transformer
- attention
- neural
- graph
- quantum
- language
- diffusion
- embedding
""")

# -----------------------------------------------------
# Similar Words
# -----------------------------------------------------

if menu == "🔍 Similar Words":

    st.header("🔍 Find Similar Research Terms")

    st.write(
        "Enter a scientific keyword to discover semantically related words."
    )

    word = st.text_input(
        "Research Keyword",
        placeholder="Example : transformer"
    )

    topn = st.slider(
        "Number of Results",
        min_value=1,
        max_value=20,
        value=10
    )

    if st.button("🚀 Search"):

        result = get_similar_words(word, topn)

        if result:

            st.success(f"Top {topn} Similar Words")

            df = pd.DataFrame(
                result,
                columns=[
                    "Similar Word",
                    "Similarity Score"
                ]
            )

            df.index = df.index + 1
            df.index.name = "Rank"

            st.dataframe(
                df,
                use_container_width=True
            )

        else:

            st.error("❌ Word not found in vocabulary.")

# -----------------------------------------------------
# Word Vector
# -----------------------------------------------------

elif menu == "🧠 Word Vector":

    st.header("🧠 View Word Embedding")

    st.write(
        "Generate the embedding vector of any research term."
    )

    word = st.text_input(
        "Research Keyword",
        placeholder="Example : neural"
    )

    if st.button("Generate Vector"):

        vector = get_vector(word)

        if vector is not None:

            st.success("Embedding Generated Successfully")

            st.metric(
                "Embedding Dimension",
                len(vector)
            )

            st.subheader("First 20 Values")

            st.code(vector[:20])

            with st.expander("View Complete Vector"):

                st.write(vector)

        else:

            st.error("❌ Word not found in vocabulary.")

# -----------------------------------------------------
# Similarity
# -----------------------------------------------------

elif menu == "📈 Word Similarity":

    st.header("📈 Semantic Similarity")

    st.write(
        "Compare the semantic relationship between two scientific terms."
    )

    col1, col2 = st.columns(2)

    with col1:
        word1 = st.text_input(
            "First Word",
            placeholder="deep"
        )

    with col2:
        word2 = st.text_input(
            "Second Word",
            placeholder="learning"
        )

    if st.button("Calculate Similarity"):

        score = similarity(word1, word2)

        if score is not None:

            st.success(f"Similarity Score : {score:.4f}")

            progress = max(0.0, min(float(score), 1.0))
            st.progress(progress)

            if score >= 0.85:
                level = "🟢 Extremely Similar"

            elif score >= 0.70:
                level = "🟢 Strong Similarity"

            elif score >= 0.50:
                level = "🟡 Moderate Similarity"

            elif score >= 0.30:
                level = "🟠 Weak Similarity"

            else:
                level = "🔴 Very Low Similarity"

            st.info(level)

        else:

            st.error("❌ One or both words are not present in the vocabulary.")

# -----------------------------------------------------
# Footer
# -----------------------------------------------------

st.divider()

st.markdown("""
### 🚀 Project Highlights

- 📚 **Dataset:** arXiv Research Papers
- 🤖 **Model:** Word2Vec
- 🧠 **Task:** Semantic Word Exploration
- 🌐 **Framework:** Streamlit
- 🐍 **Language:** Python

---
<center>

Developed for NLP Research • Word Embeddings • Semantic Search

</center>
""", unsafe_allow_html=True)