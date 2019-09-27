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

# Preciso converter para um formato que o classificador entenda.
def convert_to_features(X):
    resultados = []
    for item in X:

        tem_ltda = 0
        if "LTDA" in item:
            tem_ltda = 1

        tem_sa = 0
        if "S.A" in item:
            tem_sa = 1

        numero_palavras = item.count(' ')

        resultados.append([tem_ltda, tem_sa, numero_palavras])

    return resultados

classifier.fit(convert_to_features(X_train), y_train)

# Testes
predicted_labels = classifier.predict(convert_to_features(X_test))
target_names = ['pessoa', 'empresa']
print(classification_report(y_test, predicted_labels, target_names=target_names))
