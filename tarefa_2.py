# Ola vamos criar
import pickle
from sklearn.metrics import classification_report

# Vamos abrir o nosso dataset
with open('itens.pkl', 'rb') as f_in:
    items = pickle.load(f_in)

print(f"Carregamos um conjunto com {len(items)} itens")

metade = int(len(items)/2)
X_test, y_test = zip(*items[:metade])  # Vamos separar metade para teste
X_train, y_train = zip(*items[metade:])  # O restante vai ser usado para treino

# Função para implementar as features
def predict(textos_teste):
    resultados = []
    for item in textos_teste:
        # TODO: Fazer regras para identificar se for empresa ou nao pelo texto
        eh_empresa = True
        #print(item)
        if ("LTDA" in item):
            resultados.append("Empresa")
        elif ("SA" in item):
            resultados.append("Empresa")
        else:
            resultados.append("Pessoa")

    return resultados

# Testes
predicted_labels = predict(X_test)
target_names = ['pessoa', 'empresa']
print(classification_report(y_test, predicted_labels, target_names=target_names))
