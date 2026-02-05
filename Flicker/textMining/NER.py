import re
import unicodedata
from collections import Counter, defaultdict
import spacy

nlp = spacy.load("fr_core_news_sm")
nlp.max_length = 2_000_000

KEEP_ENT_LABELS = {"LOC", "GPE", "ORG", "MISC"}
#LOC = lieux (ex: parc de la tête d'or)
#GPE = entités géopolitiques (ex: lyon)
#ORG = organisations (ex: théâtre des célestins)
#MISC = divers (ex: fête des lumières)

# Gros génériques à virer
BLACKLIST = {
    # Géographiques trop génériques
    "france", "europe",

    # Mots passe-partout
    "ville", "quartier", "centre", "site", "photo", "image",

    # Marques téléphone
    "iphone", "apple",
    "samsung", "galaxy",
    "huawei",
    "xiaomi", "redmi",
    "oneplus",
    "oppo",
    "vivo",
    "google", "pixel",

    # Marques appareil photo
    "canon",
    "nikon",
    "sony",
    "fujifilm", "fuji",
    "panasonic", "lumix",
    "olympus", "om system",
    "leica",
    "pentax",
    "ricoh",
    "gopro",

    # Termes techniques fréquents
    "mm", "mp", "iso", "raw", "jpeg", "hdr", "lens", "objectif"
}

# caractères bizarres à supprimer
RE_WEIRD = re.compile(r"[^\w\s'’\-]", flags=re.UNICODE) #Garde seulement les lettres, chiffres, espaces, apostrophes et tirets
#enlève emoji, ponctuation et caractères spéciaux
RE_SPACES = re.compile(r"\s+") #enlève les suites d'espaces ("  " -> " ")
RE_DIGIT_LETTER = re.compile(r"(\d)([A-Za-zÀ-ÿ])") #Repère quand une lettre est collée à un chiffre, pour insérer un espace.
#"Lyon4" -> "Lyon 4"
RE_LETTER_DIGIT = re.compile(r"([A-Za-zÀ-ÿ])(\d)") #même chose dans l'autre sens, "4Lyon" -> "4 Lyon"

RE_GLUE = re.compile(r"([a-zà-öø-ÿ]{3,})([A-ZÀ-ÖØ-öø-ÿ][a-zà-öø-ÿ]{2,})") #séparer deux blocs de mots collés quand le second commence par une majuscule.
#(ex: HelloWorld -> Hello World)
RE_ALLLOWER_GLUE = re.compile(r"([a-zà-öø-ÿ]{4,})(lyon|rhone|saone)$", re.IGNORECASE)
#séparer un nom de lieu collé à "lyon", "rhone" ou "saone" (ex: CelestinsLyon -> Celestins Lyon)

def _strip_accents(s: str) -> str:
    """Enlève les accents pour faciliter le dédoublonnage. """ 
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )

def _normalize_display(s: str) -> str:
    """Nettoyage pour affichage."""
    s = s.replace("_", " ") #change les tirets en espaces
    s = s.replace("-", " ")
    s = s.replace("µ", " ") # enlève µ 
    s = RE_LETTER_DIGIT.sub(r"\1 \2", s)
    s = RE_DIGIT_LETTER.sub(r"\1 \2", s)
    s = RE_WEIRD.sub(" ", s)
    s = RE_SPACES.sub(" ", s).strip()

    # essaie de séparer quelques collages
    s = RE_GLUE.sub(r"\1 \2", s)
    s = RE_ALLLOWER_GLUE.sub(lambda m: f"{m.group(1)} {m.group(2)}", s)
    s = RE_SPACES.sub(" ", s).strip()
    return s

def _canonical(s: str) -> str:
    """Forme canonique pour dédoublonner."""
    s = _normalize_display(s).lower()
    s = _strip_accents(s)
    return s

def _is_garbage_phrase(phrase: str) -> bool:
    p = _canonical(phrase)
    if not p or len(p) < 4:
        return True

    # filtre pays/continents trop génériques
    if p in {"france", "europe"}:
        return True

    return False

def _extract_candidates(text: str):
    doc = nlp(text)
    cands = []

    # Entités
    for ent in doc.ents:
        if ent.label_ in KEEP_ENT_LABELS:  #Si l'entité est de type lieu, organisation, etc. (entité pertinente)
            cands.append(ent.text)

    # Groupes nominaux contenant un PROPN
    for chunk in doc.noun_chunks:
        if any(tok.pos_ == "PROPN" for tok in chunk):
            cands.append(chunk.text)

    return cands

def _pick_best_variant(variants):
    """
    Choisit la meilleure variante d'affichage.
    Critères (dans l'ordre) :
    1) plus longue
    2) contient des majuscules
    3) contient des accents
    """
    return max(
        variants,
        key=lambda v: (
            len(_normalize_display(v)),                    # 1) longueur
            any(c.isupper() for c in v),                  # 2) majuscule ?
            _strip_accents(v) != v                       # 3) accent ?
        )
    )

def process_texts_and_extract_keywords(texts, top_n=5):
    counter = Counter()
    variants = defaultdict(list)

    for t in texts:
        if not t or not str(t).strip():
            continue
        for cand in _extract_candidates(str(t)):
            disp = _normalize_display(cand)
            if not disp:
                continue
            if _is_garbage_phrase(disp):
                continue

            can = _canonical(disp)
            toks = set(can.split())
            if toks & BLACKLIST:   # intersection non vide
                continue

            if toks == {"lyon"}:
                continue

            counter[can] += 1
            variants[can].append(disp)

    if not counter:
        return []

    # tri: fréquence puis longueur (favorise expressions)
    ranked = sorted(counter.items(), key=lambda x: (x[1], len(x[0])), reverse=True)

    # dédoublonnage par couverture (si "theatre des celestins lyon" garde, drop "celestins lyon")
    selected_can = []
    for can, _ in ranked:
        if any(can in other for other in selected_can):   # sous-phrase d’une plus longue déjà prise
            continue
        selected_can.append(can)
        if len(selected_can) >= top_n * 3:  # marge pour filtrage final
            break

    # reconstruire sortie avec meilleure variante d'affichage + score
    out = []
    for can in selected_can:
        best = _pick_best_variant(variants[can])
        out.append((best, counter[can]))
        if len(out) >= top_n:
            break

    return out

def format_keywords(keywords, max_words=3):
    if not keywords:
        return ""
    keywords = keywords[:max_words]
    return ", ".join([kw for kw, _ in keywords])