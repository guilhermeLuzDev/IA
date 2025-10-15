# -*- coding: utf-8 -*-
"""Heart_Disease_Prediction_Colab_Script_FINISHED.py

Projetos para previsão de óbito por insuficiência cardíaca usando Streamlit
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_curve, roc_auc_score, precision_score, recall_score, f1_score)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

st.title("Previsão de Óbito por Insuficiência Cardíaca")

"""## 2. Carregamento do dataset"""
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
st.write('Formato:', df.shape)
st.dataframe(df.head())

"""## 3. Análise exploratória (EDA)"""

# Captura saída do df.info()
import io
buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()
st.text(info_str)

st.write(df.describe().T)
st.write('Valores nulos por coluna:')
st.write(df.isnull().sum())

# Distribuição do alvo (DEATH_EVENT)
plt.figure(figsize=(6,4))
sns.countplot(x='DEATH_EVENT', data=df)
plt.title('Distribuição de DEATH_EVENT (0 = vivo, 1 = óbito)')
st.pyplot(plt.gcf())
plt.clf()

# Correlação e heatmap
plt.figure(figsize=(12,10))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Mapa de Correlação')
st.pyplot(plt.gcf())
plt.clf()

"""## 4. Pré-processamento"""

# Features e target
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']
st.write('X shape:', X.shape, 'y shape:', y.shape)

# Normalização das features numéricas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
st.write('Treino:', X_train.shape, 'Teste:', X_test.shape)

"""## 5. Treinamento — modelos base (benchmark)"""

# Logistic Regression
log = LogisticRegression(max_iter=1000)
log.fit(X_train, y_train)
y_pred_log = log.predict(X_test)
acc_log = accuracy_score(y_test, y_pred_log)

# KNN (k=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)

# Random Forest (base)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

st.write('Acurácia — Logistic Regression: {:.3f}'.format(acc_log))
st.write('Acurácia — KNN: {:.3f}'.format(acc_knn))
st.write('Acurácia — Random Forest: {:.3f}'.format(acc_rf))

"""## 6. Avaliação detalhada e comparação"""

models = {'Logistic Regression': log, 'KNN': knn, 'Random Forest': rf}
results = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1]) if hasattr(model, 'predict_proba') else roc_auc_score(y_test, model.predict(X_test))
    results[name] = {'accuracy': acc, 'roc_auc': auc}
    st.write('---', name)
    st.text(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f'Confusion Matrix — {name}')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    st.pyplot(plt.gcf())
    plt.clf()

results_df = pd.DataFrame(results).T
st.dataframe(results_df)

# Curvas ROC
plt.figure(figsize=(8,6))
for name, model in models.items():
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)[:,1]
    else:
        try:
            probs = model.decision_function(X_test)
        except:
            probs = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
plt.plot([0,1],[0,1],'--', color='grey')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
st.pyplot(plt.gcf())
plt.clf()

"""## 7. Importância de features (Random Forest)"""

feat_imp = pd.Series(rf.feature_importances_, index=df.drop('DEATH_EVENT', axis=1).columns)
feat_imp = feat_imp.sort_values(ascending=False)
plt.figure(figsize=(8,6))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title('Importância das features — Random Forest')
st.pyplot(plt.gcf())
plt.clf()

"""## 8. Validação cruzada e ajuste de hiperparâmetros (exemplo)"""

# Validação cruzada com Random Forest (score médio)
cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='roc_auc')
st.write('Random Forest CV AUC (5-fold):', cv_scores.mean())

# Exemplo de GridSearch para Random Forest (rápido)
param_grid = {'n_estimators': [50,100], 'max_depth':[None,5,10]}
gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='roc_auc')
gs.fit(X_train, y_train)
st.write('Melhores parâmetros RF:', gs.best_params_)
best_rf = gs.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)
st.write('Acurácia do RF ajustado:', accuracy_score(y_test, y_pred_best_rf))

"""## 9. Melhorias: Escolhendo o melhor modelo"""

import joblib
joblib.dump(best_rf, 'best_random_forest_model.joblib')
st.write('Modelo salvo: best_random_forest_model.joblib')
