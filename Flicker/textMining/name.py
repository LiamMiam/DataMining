import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("fr_core_news_sm")
nlp.max_length = 2_000_000  # sécurité si texte un peu long


def preprocess_text(text):
    """
    Nettoyage NLP :
    - minuscule
    - enlève stop words et ponctuation
    - garde noms propres, noms communs et adjectifs
    - lemmatisation
    """
    doc = nlp(text)
    
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and token.pos_ in {"NOUN", "PROPN", "ADJ"}
        and len(token) > 2
    ]
    
    return " ".join(tokens)

def extract_top_keywords(corpus, top_n=5):
    """
    Calcule le TF-IDF sur un corpus
    et retourne les top_n mots les plus importants
    """
    vectorizer = TfidfVectorizer()

    # Calcule la matrice TF-IDF pour tout le corpus
    # Chaque ligne = un document, chaque colonne = un mot du vocabulaire
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Récupère la liste des mots correspondant aux colonnes de la matrice
    feature_names = np.array(vectorizer.get_feature_names_out())

    # Somme des scores TF-IDF pour chaque mot sur tous les documents
    # Cela donne un score global par mot dans tout le corpus
    scores = tfidf_matrix.toarray().sum(axis=0)

    # Trier par score décroissant
    top_indices = scores.argsort()[::-1][:top_n]

    # Retourne les mots et leurs scores sous forme de liste de tuples
    return list(zip(feature_names[top_indices], scores[top_indices]))


def process_texts_and_extract_keywords(texts):
    """
    Prétraite une liste de textes et extrait les mots clés
    """
    processed_texts = [preprocess_text(t) for t in texts]
    keywords = extract_top_keywords(processed_texts, top_n=5)
    return keywords

def format_keywords(keywords, max_words=3):
    """
    Transforme la liste de tuples (mot, score) en chaîne lisible pour le popup.
    """
    if not keywords:
        return ""
    
    # Garde seulement max_words mots
    keywords = keywords[:max_words]
    
    # Formattage HTML : "mot1, mot2, mot3"
    formatted = ", ".join([kw[0] for kw in keywords])
    
    return formatted