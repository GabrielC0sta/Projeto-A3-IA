import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning  
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import warnings

# Carregar o conjunto de dados
df = pd.read_csv('data.csv')

# Remover coluna não identificada
df = df.drop(['Unnamed: 32'], axis=1)

# Separar variáveis preditoras e target
X = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis']

# Dividir em colunas numéricas e categóricas
numeric_cols = X.select_dtypes(include=np.number).columns
categorical_cols = X.select_dtypes(include='object').columns

# Transformações para dados numéricos e categóricos
numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = SimpleImputer(strategy='most_frequent')

# Aplicar transformações aos dados
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numeric_transformer, numeric_cols),
        ('categorical', categorical_transformer, categorical_cols)
    ])

X = preprocessor.fit_transform(X)

# Dividir dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Padronizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lidar com avisos de convergência
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Definir o grid de hiperparâmetros para a busca
param_grid = {
    'hidden_layer_sizes': [(3, 3), (5, 5), (5, 2), (10, 5)],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [5000]  
}

# Criar o classificador MLP e realizar a busca em grade
clf = MLPClassifier(solver='lbfgs', random_state=1, max_iter=5000)

grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Mostrar os melhores parâmetros encontrados
print("Melhores parâmetros:", grid_search.best_params_)

# Obter o melhor classificador
best_clf = grid_search.best_estimator_

# Fazer previsões no conjunto de teste
prediction = best_clf.predict(X_test_scaled)

# Mostrar relatório de classificação e matriz de confusão
print('Relatório:\n', classification_report(y_test, prediction))
print("\nMatriz de confusão detalhada:\n",
      pd.crosstab(y_test, prediction, rownames=['Real'], colnames=['Predito'],
                  margins=True, margins_name='Todos'))

# Calcular e mostrar a acurácia do modelo
accuracy = best_clf.score(X_test_scaled, y_test)
print(f'Acurácia do Modelo: {accuracy:.2f}')

# Plotar a curva ROC
y_true_binary = y_test.map({'B': 0, 'M': 1})
y_scores = best_clf.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_true_binary, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
