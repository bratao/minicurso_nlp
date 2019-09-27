# Ola vamos criar
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

with open('itens.pkl', 'rb') as f_in:
    items = pickle.load(f_in)

print(f"Carregamos um conjunto com {len(items)} itens")

metade = int(len(items)/2)
X_test, y_test = zip(*items[:metade])  # Vamos separar metade para teste
X_train, y_train = zip(*items[metade:])  # O restante vai ser usado para treino


classifier = DecisionTreeClassifier(class_weight='balanced')

# this vectorizer will skip stop words
vectorizer = CountVectorizer()
# fit the vectorizer on the training text
vectorizer.fit(X_train)

X_transformado = vectorizer.transform(X_train)

classifier.fit(X_transformado, y_train)

# Testes
predicted_labels = classifier.predict(vectorizer.transform(X_test))
target_names = ['pessoa', 'empresa']
print(classification_report(y_test, predicted_labels, target_names=target_names))
