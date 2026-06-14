# =====================================================
# PIPELINE NLP
# Tokenización - Stopwords - Lematización - TF-IDF
# =====================================================

from collections import Counter

import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# FUNCIONES

def quitar_stopwords_eng(tokens):
    stop_words = set(stopwords.words("english"))

    return [
        token.lower()
        for token in tokens
        if token.isalpha()
        and token.lower() not in stop_words]


def lematizar(tokens):
    lemmatizer = WordNetLemmatizer()

    return [
        lemmatizer.lemmatize(token)
        for token in tokens]


def procesar_texto(texto):

    tokens = word_tokenize(texto)

    tokens = quitar_stopwords_eng(tokens)

    tokens = lematizar(tokens)

    return tokens

# CORPUS
corpus_original = [

    "Python is an amazing programming language.",

    "Natural Language Processing is a branch of Artificial Intelligence.",

    "Python is widely used in Machine Learning and NLP.",

    "Text Mining extracts useful information from documents.",

    "Machine Learning and NLP are transforming modern technology."]

# PROCESAMIENTO DEL CORPUS

corpus_procesado = [
    " ".join(procesar_texto(doc))
    for doc in corpus_original
]

print("\nCORPUS PROCESADO\n")

for doc in corpus_procesado:
    print(doc)
    
# ANÁLISIS DE FRECUENCIAS

tokens_totales = " ".join(corpus_procesado).split()

frecuencias = Counter(tokens_totales)

print("\nFRECUENCIA DE TÉRMINOS\n")

for palabra, frecuencia in frecuencias.most_common():
    print(f"{palabra:<15} {frecuencia}")

# ESTADÍSTICAS
print("\nESTADÍSTICAS DEL CORPUS\n")

print("Documentos       :", len(corpus_original))
print("Palabras totales :", len(tokens_totales))
print("Palabras únicas  :", len(set(tokens_totales)))
print(
    "Promedio/doc     :",
    round(len(tokens_totales) / len(corpus_original), 2))

# TF-IDF
vectorizador = TfidfVectorizer()

matriz_tfidf = vectorizador.fit_transform(corpus_procesado)

print("\nVOCABULARIO TF-IDF\n")

for termino in vectorizador.get_feature_names_out():
    print(termino)

# GRÁFICO DE FRECUENCIAS

top_10 = frecuencias.most_common(10)

palabras = [x[0] for x in top_10]
cantidades = [x[1] for x in top_10]

plt.figure(figsize=(10, 5))

plt.bar(palabras, cantidades)

plt.title("Distribución de Frecuencia de Términos")

plt.xlabel("Palabras")

plt.ylabel("Frecuencia")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()
