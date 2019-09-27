import pickle

import numpy as np
import pymysql
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

cnx = pymysql.connect(
    host='127.0.0.1',
    database='treinamento_lua',
    user='root',
    password='',
    use_unicode=True)

cursor = cnx.cursor()

cursor.execute('SELECT * FROM ' +
                '( ' +
                    '(select nome, "Empresa" as tipo from empresa) ' +
                    'union ' +
                    '(select nome, "Pessoa" as tipo from pessoa) ' +
                ') entidades order by rand()')

items = cursor.fetchall()

with open('itens.pkl', 'wb') as output:
    pickle.dump(items, output, pickle.HIGHEST_PROTOCOL)

# Teste
textos, tags = zip(*items[8000:])
textos_teste, tags_teste = zip(*items[:8000])

#textos, tags = zip(*items)

classifier = LinearSVC(loss='hinge', penalty='l2', random_state=42, class_weight = 'balanced')

clf = Pipeline([('vectorizer', CountVectorizer(strip_accents='unicode', ngram_range=(1, 2))),
    ('transformer', TfidfTransformer()),
    ('classifier', classifier)])
clf.fit(textos, tags)

# Testes
predicted_labels = clf.predict(textos_teste)
print(np.mean(predicted_labels == tags_teste))

# predicted_labels = clf.predict(textos)
# print(np.mean(predicted_labels == tags))


joblib.dump(clf, 'model_pessoa_empresa.pkl')


# clf = joblib.load('model_pessoa_empresa.pkl')
