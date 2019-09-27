import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

data = pd.DataFrame()
# Hipotese. Pessoas sem filhos, ou maiores de 50 anos sao bons pagadores
data['sexo'] =          ['m', 'm', 'm', 'f', 'f', 'f', 'f']
data['idade'] =         [30, 60, 65, 18, 25, 56, 19]
data['filhos'] =        [1, 0, 1, 0, 0, 1, 1]
data['Class'] =         [1, 0, 0, 0, 0, 0, 1]
# 0 = bom 1 = mau

tree = DecisionTreeClassifier()

tree.fit(data, data['Class'])

le = preprocessing.LabelEncoder()
# Vamos converter a string para uma categoria
le.fit(data['sexo'])

# Aplicar o encoder para a coluna
data['sexo'] = le.transform(data['sexo'])



tree.fit(data, data['Class'])




