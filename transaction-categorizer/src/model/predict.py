
import argparse, joblib, json, os
from ..preprocessing.preprocess import preprocess_text


# Load ML Models

def load_models(models_dir):
    vect = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.joblib'))
    clf = joblib.load(os.path.join(models_dir, 'calibrated_clf.joblib'))
    le = joblib.load(os.path.join(models_dir, 'label_encoder.joblib'))
    return vect, clf, le

# Load Rule-Based Keywords

def load_rules():
    possible_paths = [
        os.path.join(os.getcwd(), "categories.json"),
        os.path.join(os.getcwd(), "config", "categories.json"),
        os.path.join(os.path.dirname(__file__), "..", "categories.json"),
        os.path.join(os.path.dirname(__file__), "..", "config", "categories.json")
    ]
    for path in possible_paths:
        path = os.path.abspath(path)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    raise FileNotFoundError("categories.json not found in any known location.")

# Rule-Based Override Check

def rule_based_category(preprocessed_text, rules):
    for category, keywords in rules.items():
        for kw in keywords:
            kw_pre = preprocess_text(kw)  
            if kw_pre in preprocessed_text:  
                return category
    return None

# Predict Function

def predict_single(text, models_dir='models/'):
    vect, clf, le = load_models(models_dir)
    rules = load_rules()

    pre = preprocess_text(text)

    #  RULE BASED OVERRIDE 

    rule_cat = rule_based_category(pre, rules)
    if rule_cat is not None:
        return {
            "input": text,
            "preprocessed": pre,
            "predicted": rule_cat,
            "confidence": 1.0,
            "probs": {cat: (1.0 if cat == rule_cat else 0.0) for cat in rules.keys()}
        }

    #  ML PREDICTION 
    X = vect.transform([pre])
    probs = clf.predict_proba(X)[0]
    idx = probs.argmax()

    return {
        'input': text,
        'preprocessed': pre,
        'predicted': str(le.inverse_transform([idx])[0]),
        'confidence': float(probs[idx]),
        'probs': {str(le.inverse_transform([i])[0]): float(p) for i, p in enumerate(probs)}
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', required=True)
    parser.add_argument('--models', default='models/')
    args = parser.parse_args()

    result = predict_single(args.text, args.models)

    print("\n================ PREDICTION RESULT ================\n")
    print(f"Input: {result['input']}")
    print(f"Preprocessed: {result['preprocessed']}")
    print(f"Predicted Category: {result['predicted']}")
    print(f"Confidence: {round(result['confidence'], 4)}\n")

    print("Category Probabilities:")
    for cat, prob in result["probs"].items():
        print(f" - {cat}: {round(prob, 4)}")

    print("\n===================================================\n")
