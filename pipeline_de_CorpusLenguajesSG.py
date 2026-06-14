# ==========================================================
# INFORME DE CORPUS NLP
# Tokenización - Stopwords - Lematización - TF-IDF
# ==========================================================

from collections import Counter
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer


# ==========================================================
# FUNCIONES
# ==========================================================

def quitar_stopwords_eng(tokens):
    """
    Elimina stopwords y caracteres no alfabéticos.
    """

    try:

        stop_words = set(stopwords.words("english"))

        return [

            token.lower()

            for token in tokens

            if token.isalpha()
            and token.lower() not in stop_words

        ]

    except Exception as e:

        print("Error cargando stopwords:", e)

        return tokens


def lematizar(tokens):
    """
    Lematización del texto.
    """

    try:

        lemmatizer = WordNetLemmatizer()

        return [

            lemmatizer.lemmatize(token)

            for token in tokens

        ]

    except Exception as e:

        print("Error durante la lematización:", e)

        return tokens


def procesar_texto(texto):
    """
    Pipeline completo NLP.
    """

    try:

        tokens = word_tokenize(texto)

        tokens = quitar_stopwords_eng(tokens)

        tokens = lematizar(tokens)

        return tokens

    except Exception as e:

        print("Error procesando texto:", e)

        return []


# ==========================================================
# CORPUS
# ==========================================================

corpus_original = [

    "Python is an amazing programming language.",

    "Natural Language Processing is a branch of Artificial Intelligence.",

    "Python is widely used in Machine Learning and NLP.",

    "Text Mining extracts useful information from documents.",

    "Machine Learning and NLP are transforming modern technology."

]


# ==========================================================
# PROCESAMIENTO DEL CORPUS
# ==========================================================

corpus_procesado = []

print("\n========== CORPUS PROCESADO ==========\n")

for documento in corpus_original:

    resultado = procesar_texto(documento)

    corpus_procesado.append(" ".join(resultado))

    print(resultado)


# ==========================================================
# FRECUENCIA DE TÉRMINOS
# ==========================================================

tokens_totales = []

for documento in corpus_procesado:

    tokens_totales.extend(documento.split())

frecuencias = Counter(tokens_totales)

print("\n========== FRECUENCIAS ==========\n")

for palabra, frecuencia in frecuencias.most_common():

    print(f"{palabra:<20}{frecuencia}")


# ==========================================================
# ESTADÍSTICAS DEL CORPUS
# ==========================================================

cantidad_documentos = len(corpus_original)

cantidad_palabras = len(tokens_totales)

cantidad_unicas = len(set(tokens_totales))

promedio = cantidad_palabras / cantidad_documentos

print("\n========== ESTADÍSTICAS ==========\n")

print("Documentos:", cantidad_documentos)

print("Palabras:", cantidad_palabras)

print("Palabras únicas:", cantidad_unicas)

print("Promedio por documento:", round(promedio, 2))


# ==========================================================
# TF-IDF
# ==========================================================

try:

    vectorizador = TfidfVectorizer()

    matriz_tfidf = vectorizador.fit_transform(corpus_procesado)

    vocabulario = vectorizador.get_feature_names_out()

    print("\n========== VOCABULARIO TF-IDF ==========\n")

    for termino in vocabulario:

        print(termino)

except Exception as e:

    print("Error calculando TF-IDF:", e)


# ==========================================================
# GRÁFICO DE FRECUENCIAS
# ==========================================================

try:

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

except Exception as e:

    print("Error generando gráfico:", e)
