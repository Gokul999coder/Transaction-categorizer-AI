
import argparse, os, joblib, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from ..preprocessing.preprocess import preprocess_text

def load_combined(data_path, feedback_path):
    df = pd.read_csv(data_path)
    df['raw_text'] = df[['merchant','description']].fillna('').agg(' '.join,axis=1)
    df['text'] = df['raw_text'].apply(preprocess_text)
    if os.path.exists(feedback_path):
        fb = pd.read_csv(feedback_path)
        fb['raw_text'] = fb[['merchant','description']].fillna('').agg(' '.join,axis=1)
        fb['text'] = fb['raw_text'].apply(preprocess_text)
        fb = fb.rename(columns={'correct':'label'})
        df = pd.concat([df[['merchant','description','text','label']], fb[['merchant','description','text','label']]], ignore_index=True)
    return df[df['text'].str.strip()!='']

def retrain(data_path, feedback_path='data/feedback.csv', models_dir='models/'):
    os.makedirs(models_dir, exist_ok=True)
    df = load_combined(data_path, feedback_path)
    X = df['text'].values
    y = df['label'].values
    le = LabelEncoder(); y_enc = le.fit_transform(y)
    vect = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9)
    Xv = vect.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, class_weight='balanced', solver='saga')
    clf.fit(Xv, y_enc)
    cal = CalibratedClassifierCV(clf, cv='prefit', method='isotonic')
    cal.fit(Xv, y_enc)
    joblib.dump(vect, os.path.join(models_dir,'tfidf_vectorizer.joblib'))
    joblib.dump(cal, os.path.join(models_dir,'calibrated_clf.joblib'))
    joblib.dump(le, os.path.join(models_dir,'label_encoder.joblib'))
    print("Retrained and saved models")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--feedback', default='data/feedback.csv')
    parser.add_argument('--models', default='models/')
    args = parser.parse_args()
    retrain(args.data, args.feedback, args.models)
