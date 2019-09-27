# Ola vamos criar
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

with open('itens.pkl', 'rb') as f_in:
    items = pickle.load(f_in)

print(f"Carregamos um conjunto com {len(items)} itens")

metade = int(len(items)/2)
X_test, y_test = zip(*items[:metade])  # Vamos separar metade para teste
X_train, y_train = zip(*items[metade:])  # O restante vai ser usado para treino


classifier = LinearSVC(loss='hinge', penalty='l2', random_state=42, class_weight='balanced')

clf = Pipeline([('vectorizer', CountVectorizer(strip_accents='unicode', ngram_range=(1, 2))),
                ('transformer', TfidfTransformer()),
                ('classifier', classifier)])
clf.fit(X_train, y_train)

# Testes
predicted_labels = clf.predict(X_test)
target_names = ['pessoa', 'empresa']
print(classification_report(y_test, predicted_labels, target_names=target_names))
