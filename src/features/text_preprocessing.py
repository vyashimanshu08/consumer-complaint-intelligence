import re

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


# NLTK Resources
# ==========================================

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)


# Step 2 : Stopword Configuration
# English stopwords
stop_words = set(stopwords.words("english"))

# Words that should be preserved because they can change
# the meaning of a consumer complaint.
preserve_words = {
    "not",
    "no",
    "never",
    "neither",
    "nor",
    "without",
    "against",
    "before",
    "after",
    "under",
    "over",
    "between",
    "during"
}

# Remove preserved words from the default stopword list
custom_stop_words = stop_words - preserve_words

# Step 3 — Contraction Map
# Contraction mapping
contraction_map = {
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "mustn't": "must not",
}

def expand_contractions(text: str) -> str:
    """
    Expand common English contractions into their full forms.
    """
    if not isinstance(text, str):
        return ""

    for contraction, replacement in contraction_map.items():
        text = text.replace(contraction, replacement)

    return text


# Step 4 — Lowercasing + Tokenization

#1. Lowecasing 

def lowercase_text(text: str) -> str:
    """
    Convert text to lowercase.
    """
    if not isinstance(text, str):
        return ""

    if not text.strip():
        return text

    return text.lower()

#2. Tokenization

# Tokenization pattern
_TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9]+\b")


def tokenize_text(text: str, min_token_length: int = 1) -> list[str]:
    """
    Convert text into individual tokens.
    """

    if not isinstance(text, str):
        return []

    if not text.strip():
        return []

    tokens = _TOKEN_PATTERN.findall(text)

    if min_token_length > 1:
        tokens = [
            token for token in tokens
            if len(token) >= min_token_length
        ]

    return tokens

# Remove Redaction cfbp (xxxx)

def remove_redaction_tokens(tokens: list[str]) -> list[str]:
    """
    Remove CFPB redaction placeholders such as:
    xx, xxxx, xxxxx, xxxxxx, etc.
    """

    return [
        token
        for token in tokens
        if not re.fullmatch(r"x{2,}", token)
    ]





# Removal of Stopwords 

def remove_stopwords(tokens: list[str], stopword_set: set[str]) -> list[str]:
    """
    Remove stopwords while preserving custom important words.
    """

    if not tokens:
        return []

    if not isinstance(stopword_set, set):
        stopword_set = set(stopword_set)

    return [
        word
        for word in tokens
        if word not in stopword_set
    ]



# Step 7 — POS-Aware Lemmatization
_lemmatizer = WordNetLemmatizer() # Adding lemmatizer instance

def get_wordnet_pos(nltk_tag: str) -> str:
    """
    Convert NLTK POS tags to WordNet POS tags.
    """

    if nltk_tag.startswith("J"):
        return wordnet.ADJ

    elif nltk_tag.startswith("V"):
        return wordnet.VERB

    elif nltk_tag.startswith("R"):
        return wordnet.ADV

    else:
        return wordnet.NOUN

def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """
    Lemmatize tokens using POS-aware lemmatization.
    """

    if not tokens:
        return []

    tagged_tokens = pos_tag(tokens)

    return [
        _lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in tagged_tokens
    ]

# test 3 
if __name__ == "__main__":
    test_tokens = [
        "payments",
        "reported",
        "accounts",
        "issues",
        "complaints",
        "disputes",
        "involving",
        "removed",
        "companies",
        "transactions"
    ]
 
    print("Original tokens:")
    print(test_tokens)

    print("\nLemmatized tokens:")
    print(lemmatize_tokens(test_tokens))



# Step 8 — Build the Complete clean_narrative() Function
def clean_narrative(text: str) -> str:
    """
    Complete NLP preprocessing pipeline for a consumer complaint narrative.

    Steps:
    1. Lowercase
    2. Expand contractions
    3. Tokenize
    4. Remove CFPB redaction tokens
    5. Remove custom stopwords
    6. Lemmatize
    """

    text = lowercase_text(text)

    text = expand_contractions(text)

    tokens = tokenize_text(text)

    tokens = remove_redaction_tokens(tokens)

    tokens = remove_stopwords(tokens, custom_stop_words)

    tokens = lemmatize_tokens(tokens)

    return " ".join(tokens)


