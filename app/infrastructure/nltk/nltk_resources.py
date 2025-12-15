# app/infrastructure/nltk_resources.py
import os
import nltk

def ensure_nltk_punkt():
    # optional: pin where data is stored
    os.environ.setdefault("NLTK_DATA", os.path.join(os.getcwd(), ".nltk_data"))

    # Newer NLTK may require BOTH
    for res, path in [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(res, quiet=True)
