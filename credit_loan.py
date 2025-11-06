# Manipulation de données
import pandas as pd
import numpy as np

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Préparation des données
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Modélisation
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Évaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sauvegarde
import pickle

# Réglage d'affichage
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (8,5)

# Charger le jeu de données
df = pd.read_csv("credit_loan.csv")
# Aperçu
df.head()

# Informations générales
df.info()
df.columns

# Statistiques de base
df.describe()

# Separer les variables categorielles et celles mumeriques
cat_cols = []
num_cols = []

for col in df.columns:
    if df[col].dtype == 'object':
        cat_cols.append(col)
    else:
        num_cols.append(col)

cat_data = df[cat_cols]
num_data = df[num_cols]

# Remplacer les valeurs manquantes par les valeurs qui se repetent
cat_data = cat_data.apply(lambda x: x.fillna(x.value_counts().index[0]))
cat_data.isnull().sum().any()

#Pour les variables numeriques, remplacer les valeurs manquantes par la valeur precedente dela meme colonne
num_data.fillna(method='bfill', inplace=True)
num_data.isnull().sum().any()

data = pd.concat([cat_data, num_data], axis = 1)

# Encodage des variables catégorielles
for col in data.select_dtypes('object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    
data = data.drop('Loan_ID', axis=1)

# Séparer X et y
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

# Séparer les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Tester plusieurs modèles
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results.append({'Modèle': name, 'Précision': acc})
    print(f"{name}: {acc*100:.2f}%")

pd.DataFrame(results)

best_model = RandomForestClassifier(n_estimators=100, random_state=42)
best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)

print(classification_report(y_test, y_pred))
# sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
# plt.title('Matrice de confusion')
# plt.show()

# Vérifions la répartition des classes
df['Loan_Status'].value_counts(normalize=True) * 100

from imblearn.over_sampling import SMOTE

# Création d’un objet SMOTE
sm = SMOTE(random_state=42)

# Rééchantillonnage uniquement sur le jeu d'entraînement
X_res, y_res = sm.fit_resample(X_train_scaled, y_train)

# Vérifions la nouvelle répartition
print("Avant SMOTE :", y_train.value_counts().to_dict())
print("Après SMOTE :", pd.Series(y_res).value_counts().to_dict())

# Réentraîner le même modèle (Logistic Regression)
balanced_model = LogisticRegression(random_state=42, max_iter=500)
balanced_model.fit(X_res, y_res)

# Prédiction sur le jeu de test (non équilibré)
y_pred_balanced = balanced_model.predict(X_test_scaled)

# Évaluation
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred_balanced))
sns.heatmap(confusion_matrix(y_test, y_pred_balanced), annot=True, fmt='d', cmap='Blues')
plt.title("Matrice de confusion après SMOTE")
plt.show()


from sklearn.metrics import f1_score, recall_score, precision_score

# Avant SMOTE
y_pred_base = best_model.predict(X_test_scaled)
f1_before = f1_score(y_test, y_pred_base)
recall_before = recall_score(y_test, y_pred_base)

# Après SMOTE
f1_after = f1_score(y_test, y_pred_balanced)
recall_after = recall_score(y_test, y_pred_balanced)

comparison = pd.DataFrame({
    "Métrique": ["F1-score", "Recall"],
    "Avant SMOTE": [f1_before, recall_before],
    "Après SMOTE": [f1_after, recall_after]
})
comparison

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# Définition de la grille de recherche
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs'],
    'class_weight': [None, 'balanced']
}

# Création de l’objet GridSearch
grid_lr = GridSearchCV(
    LogisticRegression(random_state=42, max_iter=500),
    param_grid_lr,
    scoring='f1',
    cv=5,
    n_jobs=-1
)

# Entraînement sur les données équilibrées
grid_lr.fit(X_res, y_res)

print(" Meilleurs hyperparamètres :", grid_lr.best_params_)
print(" Meilleur score F1 (validation croisée) :", round(grid_lr.best_score_, 3))

# Prédictions avec le meilleur modèle
best_lr = grid_lr.best_estimator_
y_pred_best = best_lr.predict(X_test_scaled)

# Rapport de classification
print(classification_report(y_test, y_pred_best))

# Matrice de confusion
sns.heatmap(confusion_matrix(y_test, y_pred_best), annot=True, fmt='d', cmap='Greens')
plt.title("Matrice de confusion - Logistic Regression optimisée")
plt.show()

from sklearn.ensemble import RandomForestClassifier

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 8, None],
    'min_samples_split': [2, 5, 10],
    'class_weight': [None, 'balanced']
}

grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    scoring='f1',
    cv=5,
    n_jobs=-1
)

grid_rf.fit(X_res, y_res)

print("Meilleurs hyperparamètres RF :", grid_rf.best_params_)
print("Meilleur score F1 (validation croisée) :", round(grid_rf.best_score_, 3))

# Évaluation sur le jeu de test
y_pred_rf = grid_rf.best_estimator_.predict(X_test_scaled)
print(classification_report(y_test, y_pred_rf))

results = pd.DataFrame({
    "Modèle": ["Logistic Regression (optimisée)", "Random Forest (optimisé)"],
    "F1 Validation": [grid_lr.best_score_, grid_rf.best_score_],
    "F1 Test": [
        f1_score(y_test, y_pred_best),
        f1_score(y_test, y_pred_rf)
    ],
    "Recall Test": [
        recall_score(y_test, y_pred_best),
        recall_score(y_test, y_pred_rf)
    ],
    "Accuracy Test": [
        accuracy_score(y_test, y_pred_best),
        accuracy_score(y_test, y_pred_rf)
    ]
})
results

import joblib

# sauver le modèle au format joblib
joblib.dump(best_lr, 'model_credit.joblib')
