
import argparse, os, joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from sklearn.calibration import CalibratedClassifierCV

# local import
from ..preprocessing.preprocess import preprocess_text

def load_and_prep(path):
    df = pd.read_csv(path)
    df['raw_text'] = df[['merchant','description']].fillna('').agg(' '.join,axis=1)
    df['text'] = df['raw_text'].apply(preprocess_text)
    df = df[df['text'].str.strip()!='']
    return df

def train(data_path, models_dir, grid=True):
    os.makedirs(models_dir, exist_ok=True)
    df = load_and_prep(data_path)
    X = df['text'].values
    y = df['label'].values
    le = LabelEncoder(); y_enc = le.fit_transform(y)
    Xtr, Xte, ytr, yte = train_test_split(X,y_enc,test_size=0.2,random_state=42,stratify=y_enc)

    vect = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9)
    Xtr_v = vect.fit_transform(Xtr); Xte_v = vect.transform(Xte)

    base = LogisticRegression(max_iter=2000, class_weight='balanced', solver='saga')
    if grid:
        gs = GridSearchCV(base, param_grid={'C':[0.1,0.5,1.0,5.0]}, cv=3, scoring='f1_macro', n_jobs=-1)
        gs.fit(Xtr_v, ytr)
        base = gs.best_estimator_
        print("GridSearch params:", gs.best_params_)
    else:
        base.fit(Xtr_v, ytr)

    # calibrate
    cal = CalibratedClassifierCV(base, cv='prefit', method='isotonic')
    cal.fit(Xtr_v, ytr)

    preds = cal.predict(Xte_v)
    print(classification_report(yte,preds, target_names=le.classes_))
    print("Macro F1:", f1_score(yte,preds,average='macro'))

    joblib.dump(vect, os.path.join(models_dir,'tfidf_vectorizer.joblib'))
    joblib.dump(cal, os.path.join(models_dir,'calibrated_clf.joblib'))
    joblib.dump(le, os.path.join(models_dir,'label_encoder.joblib'))
    print("Saved models to", models_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--models', default='models/')
    parser.add_argument('--no-grid', action='store_true')
    args = parser.parse_args()
    train(args.data, args.models, grid=not args.no_grid)
