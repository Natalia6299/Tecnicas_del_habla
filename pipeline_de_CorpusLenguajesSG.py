# === PIPELINE NLP ===
# Tokenización -> Stopwords -> Lematización

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# (ejecutar solo la primera vez)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


# función para quitar stopwords (inglés)
def quitarStopwords_eng(tokens):
    stop_words = set(stopwords.words('english'))
    return [t for t in tokens if t.isalpha() and t.lower() not in stop_words]


# función de lematización
def lematizar(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]


# === CORPUS (dado por el ejercicio) ===
corpus = [
    lematizar(quitarStopwords_eng(word_tokenize("Python is an interpreted and high-level language, while CPlus is a compiled and low-level language .-"))),
    lematizar(quitarStopwords_eng(word_tokenize("JavaScript runs in web browsers, while Python is used in various applications, including data science and artificial intelligence."))),
    lematizar(quitarStopwords_eng(word_tokenize("JavaScript is dynamically and weakly typed, while Rust is statically typed and ensures greater data security .-"))),
    lematizar(quitarStopwords_eng(word_tokenize("Python and JavaScript are interpreted languages, while Java, CPlus, and Rust require compilation before execution."))),
    lematizar(quitarStopwords_eng(word_tokenize("JavaScript is widely used in web development, while Go is ideal for servers and cloud applications."))),
    lematizar(quitarStopwords_eng(word_tokenize("Python is slower than CPlus and Rust due to its interpreted nature."))),
    lematizar(quitarStopwords_eng(word_tokenize("JavaScript has a strong ecosystem with Node.js for backend development, while Python is widely used in data science .-"))),
    lematizar(quitarStopwords_eng(word_tokenize("JavaScript does not require compilation, while CPlus and Rust require code compilation before execution .-"))),
    lematizar(quitarStopwords_eng(word_tokenize("Python and JavaScript have large communities and an extensive number of available libraries."))),
    lematizar(quitarStopwords_eng(word_tokenize("Python is ideal for beginners, while Rust and CPlus are more suitable for experienced programmers.")))
]


# === MOSTRAR RESULTADOS ===
print("=== RESULTADO DEL PIPELINE ===\n")

for i, doc in enumerate(corpus):
    print(f"Documento {i+1}:")
    print(doc)
  
