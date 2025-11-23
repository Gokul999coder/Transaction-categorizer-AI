
import argparse, joblib, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from ..preprocessing.preprocess import preprocess_text

def load_prep(path):
    df = pd.read_csv(path)
    df['raw_text'] = df[['merchant','description']].fillna('').agg(' '.join,axis=1)
    df['text'] = df['raw_text'].apply(preprocess_text)
    return df[df['text'].str.strip()!='']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--models', default='models/')
    args = parser.parse_args()
    df = load_prep(args.data)
    vect = joblib.load(args.models + 'tfidf_vectorizer.joblib')
    clf = joblib.load(args.models + 'calibrated_clf.joblib')
    le = joblib.load(args.models + 'label_encoder.joblib')
    X = vect.transform(df['text'])
    y_true = le.transform(df['label'])
    preds = clf.predict(X)
    print('Classification report:')
    print(classification_report(y_true,preds, target_names=le.classes_))
    print('Macro F1:', f1_score(y_true,preds, average='macro'))
    print('Confusion matrix:')
    print(confusion_matrix(y_true,preds))
