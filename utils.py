from gensim.models import Word2Vec

model = Word2Vec.load("word2vec.model")

def get_similar_words(word, topn=10):

    if word in model.wv:
        return model.wv.most_similar(word, topn=topn)

    return None


def get_vector(word):

    if word in model.wv:
        return model.wv[word]

    return None


def similarity(word1, word2):

    if word1 in model.wv and word2 in model.wv:
        return model.wv.similarity(word1, word2)

    return None