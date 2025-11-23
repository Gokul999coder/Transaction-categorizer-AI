
import re
import unicodedata
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords



lemmatizer = WordNetLemmatizer()
STOP = set(stopwords.words('english'))
NOISE_TOKENS = {"pos","atm","debit","credit","payment","online","purchase","txn","transfer","ref"}

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize('NFKD', s)
    s = s.lower()
    s = re.sub(r'https?://\S+',' ',s)
    s = re.sub(r'\bwww\.[^\s]+',' ',s)
    s = re.sub(r'\.com\b',' ',s)
    s = re.sub(r'[^a-z0-9 ]',' ',s)
    s = re.sub(r'\s+',' ',s).strip()
    return s

def merchant_normalize(s: str) -> str:
    s = re.sub(r'store\s*#?\s*\d+',' ',s)
    s = re.sub(r'pos\s*\d+',' ',s)
    s = re.sub(r'\bbranch\s*\d+',' ',s)
    s = re.sub(r'\s+',' ',s).strip()
    return s

def preprocess_text(s: str) -> str:
    s = normalize_text(s)
    s = merchant_normalize(s)
    tokens = [t for t in s.split() if t not in STOP and t not in NOISE_TOKENS and not t.isdigit()]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)
