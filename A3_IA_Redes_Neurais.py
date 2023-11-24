import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning  
import warnings

df = pd.read_csv('data.csv')

df = df.drop(['Unnamed: 32'], axis=1)

X = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis']

numeric_cols = X.select_dtypes(include=np.number).columns
categorical_cols = X.select_dtypes(include='object').columns

numeric_transformer = ColumnTransformer(
    transformers=[
        ('numeric', SimpleImputer(strategy='mean'), numeric_cols),
    ])

categorical_transformer = ColumnTransformer(
    transformers=[
        ('categorical', SimpleImputer(strategy='most_frequent'), categorical_cols),
    ])

preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numeric_transformer, numeric_cols),
        ('categorical', categorical_transformer, categorical_cols)
    ])

X = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

param_grid = {
    'hidden_layer_sizes': [(3, 3), (5, 5), (5, 2), (10, 5)],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [5000]  
}

clf = MLPClassifier(solver='lbfgs', random_state=1, max_iter=5000)

grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

print("parâmetros:", grid_search.best_params_)

best_clf = grid_search.best_estimator_

prediction = best_clf.predict(X_test_scaled)

from sklearn.metrics import classification_report
print('Relatório:\n',classification_report(y_test,prediction))
print("\nMatriz de confusão detalhada:\n",
      pd.crosstab(y_test, prediction, rownames=['Real'], colnames=['Predito'],
margins=True, margins_name='Todos'))

accuracy = best_clf.score(X_test_scaled, y_test)
print(f'Acurácia do Modelo depois das melhorias: {accuracy:.2f}')

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

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










