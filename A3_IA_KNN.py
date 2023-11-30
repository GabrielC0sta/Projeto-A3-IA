import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA

# Carrega o dataset
df = pd.read_csv('data.csv')

# Remove colunas 'id' e 'Unnamed: 32'; transforma 'diagnosis' em valores numéricos
df = df.drop(['id', 'Unnamed: 32'], axis=1)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Variáveis preditoras e target
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Normaliza os dados
scaler = Normalizer()
X_normalized = scaler.fit_transform(X)

# Reduz a dimensionalidade com PCA para 10 componentes
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_normalized)

# Divisão dos dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_pca, y)

# Definição dos parâmetros a serem testados para o modelo KNN
param_grid = {'n_neighbors': range(1, 21)}

modeloKNN = KNeighborsClassifier()

# Realiza a busca em grade para encontrar os melhores parâmetros
grid_search = GridSearchCV(modeloKNN, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Obtém os melhores parâmetros encontrados
melhores_parametros = grid_search.best_params_
print("Melhores parâmetros:", melhores_parametros)

# Obtém o melhor modelo com os melhores parâmetros encontrados
melhor_modelo = grid_search.best_estimator_

# Treina o modelo com o conjunto de dados reduzido por PCA
melhor_modelo.fit(X_pca, y)

# Avalia o modelo final nos dados de treinamento (utilizando todo o conjunto)
precisao = melhor_modelo.score(X_pca, y)
print("Precisão do modelo final (conjunto completo): {:.2f}%".format(precisao * 100))
