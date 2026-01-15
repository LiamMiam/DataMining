import spacy

# Charger le modèle français
nlp = spacy.load("fr_core_news_sm")

text = "Le chat qui avait faim a rapidement chassé les souris dans la vieille grange."

doc = nlp(text) # PoS tagging

for token in doc: 
    print(f"Token: {token.text}, POS Tag: {token.pos_}")   

# Tokens without stop words
tokens_no_stop = [token.text for token in doc if not token.is_stop]
print("Tokens without stop words:", tokens_no_stop)




def preprocess_text(text):
    """
    Nettoyage NLP :
    - minuscule
    - enlève stop words
    - garde noms, verbes, adjectifs
    - lemmatisation
    """
    doc = nlp(str(text))
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and token.pos_ in {"NOUN", "PROPN", "ADJ"}
    ]
    return " ".join(tokens)
