from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

# features (1 sim, 0 não)

# pelo longo?
# perna curta?
# faz auau?
puddle = [1, 1, 1]
pug = [0, 1, 0]
golden = [1, 0, 1]

gato_1 = [1, 1, 0]
gato_2 = [1, 0, 0]
gato_3 = [1, 1, 0]

# 1 => cachorro, 0 => gato
treino_x = [puddle, pug, golden, gato_1, gato_2, gato_3]
treino_y = [1, 1, 1, 0, 0, 0]  # labels / etiqueta

# Como Linear tree
model = LinearSVC()
model.fit(treino_x, treino_y)

# Fazer a predicao
viralata = [0, 0, 1]  #
# Resultado Linear
resultado_linear = model.predict([viralata])
print(f"Resultado pelo classificador Linear {resultado_linear}")

# Como decision tree
dt_bin_clf = DecisionTreeClassifier(criterion='entropy', max_depth=3,
                                    random_state=0)
# TODO: Fazer o predict pelo DecisionTreeClassifier
resultado_decision_tree = None
print(f"Resultado pelo Decision tree {resultado_decision_tree}")

