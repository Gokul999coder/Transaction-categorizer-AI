
import joblib, numpy as np

def top_features(models_dir='models/', topn=12):
    vect = joblib.load(models_dir + 'tfidf_vectorizer.joblib')
    clf = joblib.load(models_dir + 'calibrated_clf.joblib')
    le = joblib.load(models_dir + 'label_encoder.joblib')
    base = getattr(clf, 'base_estimator', clf)
    features = np.array(vect.get_feature_names_out())
    coefs = base.coef_
    res = {}
    for i,cls in enumerate(le.classes_):
        idx = np.argsort(coefs[i])[-topn:][::-1]
        res[cls] = features[idx].tolist()
    return res

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default='models/')
    args = parser.parse_args()
    print(top_features(args.models))
