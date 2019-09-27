# Ola vamos criar
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

with open('itens.pkl', 'rb') as f_in:
    items = pickle.load(f_in)

print(f"Carregamos um conjunto com {len(items)} itens")

X_total, y_total = zip(*items)  # O restante vai ser usado para treino


classifier = LinearSVC(loss='hinge', penalty='l2', random_state=42, class_weight='balanced')

clf = Pipeline([('vectorizer', CountVectorizer(strip_accents='unicode', ngram_range=(1, 2))),
                ('transformer', TfidfTransformer()),
                ('classifier', classifier)])

cv_results = cross_validate(clf, X_total, y_total, n_jobs=4, verbose=1, cv=3)

print(cv_results)
