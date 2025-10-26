# -*- coding: utf-8 -*-
import io
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score, precision_score, recall_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

# Configuração da página
st.set_page_config(page_title="Heart Failure - ML", layout="wide")
st.title("Previsão de Óbito por Insuficiência Cardíaca (Heart Failure)")

# ---------------------------
# Informações originais do notebook
# ---------------------------
about_md = """
# Projeto – Previsão de Óbito por Insuficiência Cardíaca (Heart Failure)

Notebook profissional e completo para pré-processamento, análise exploratória, treinamento e avaliação de modelos (*Regressão Logística, KNN e Random Forest*). O dataset usado é `heart_failure_clinical_records_dataset.csv`.

Objetivos:
- Entender o dataset
- Pré-processar e preparar os dados
- Treinar e comparar modelos
- Avaliar com métricas robustas e visualizar resultados

## 0. Entendendo o Dataset — Heart Failure Clinical Records

O conjunto de dados Heart Failure Clinical Records Dataset é amplamente utilizado em projetos de machine learning voltados à análise e previsão do risco de insuficiência cardíaca.  
Ele contém 299 registros clínicos de pacientes e 12 variáveis que descrevem características demográficas, hábitos de vida e medições laboratoriais, além da variável alvo que indica o óbito do paciente.

Cada linha representa um paciente, e cada coluna corresponde a uma característica médica coletada durante o acompanhamento.

---

### Atributos do Dataset:

| Coluna | Descrição | Tipo |
|:--|:--|:--|
| age | Idade do paciente (anos) | Numérico |
| anaemia | Presença de anemia (1 = sim, 0 = não) | Binário |
| creatinine_phosphokinase | Nível da enzima CPK no sangue (mcg/L) | Numérico |
| diabetes | Presença de diabetes (1 = sim, 0 = não) | Binário |
| ejection_fraction | Percentual de sangue bombeado a cada contração do coração (%) | Numérico |
| high_blood_pressure | Hipertensão (1 = sim, 0 = não) | Binário |
| platelets | Contagem de plaquetas no sangue (kiloplatelets/mL) | Numérico |
| serum_creatinine | Nível de creatinina sérica (mg/dL) | Numérico |
| serum_sodium | Nível de sódio no sangue (mEq/L) | Numérico |
| sex | Sexo biológico (1 = homem, 0 = mulher) | Binário |
| smoking | Hábito de fumar (1 = sim, 0 = não) | Binário |
| time | Tempo de acompanhamento (dias) | Numérico |
| DEATH_EVENT | Óbito durante o período de observação (1 = sim, 0 = não) | Binário |

---

### Objetivo da Análise

O principal objetivo é prever a probabilidade de morte (DEATH_EVENT) com base nas demais variáveis clínicas, utilizando modelos de machine learning supervisionado.  
Essa previsão pode auxiliar profissionais da saúde na identificação precoce de pacientes de alto risco, contribuindo para decisões médicas mais assertivas e personalizadas.
"""
st.markdown(about_md)

# ---------------------------
# 2. Carregamento do dataset
# ---------------------------
CSV_PATH = "heart_failure_clinical_records_dataset.csv"
try:
    df = pd.read_csv(CSV_PATH)
except Exception as e:
    st.error(f"Não foi possível carregar o CSV '{CSV_PATH}'. Erro: {e}")
    st.stop()

st.subheader("Amostra do dataset")
st.dataframe(df.head())

# df.info() imprime no stdout; capturamos em buffer para exibir
buf = io.StringIO()
df.info(buf=buf)
st.subheader("Info do DataFrame")
st.text(buf.getvalue())

st.subheader("Estatísticas descritivas")
st.dataframe(df.describe().T)

st.subheader("Valores nulos por coluna")
st.write(df.isnull().sum())

# ---------------------------
# 3. Análise exploratória (EDA)
# ---------------------------
st.subheader("Distribuição do alvo (DEATH_EVENT)")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x='DEATH_EVENT', data=df, ax=ax)
ax.set_title('Distribuição de DEATH_EVENT (0 = vivo, 1 = óbito)')
st.pyplot(fig, clear_figure=True)

st.subheader("Mapa de correlação")
fig, ax = plt.subplots(figsize=(12, 10))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
ax.set_title('Mapa de Correlação')
st.pyplot(fig, clear_figure=True)

# ---------------------------
# 4. Pré-processamento
# ---------------------------
if 'DEATH_EVENT' not in df.columns:
    st.error("Coluna 'DEATH_EVENT' não encontrada no dataset.")
    st.stop()

X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# Split (estratificado) e normalização (fit no treino, transform no teste)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

st.write(f"Treino: {X_train.shape} | Teste: {X_test.shape}")

# ---------------------------
# 5. Treinamento — modelos base
# ---------------------------
log = LogisticRegression(max_iter=1000)
knn = KNeighborsClassifier(n_neighbors=5)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

log.fit(X_train, y_train)
knn.fit(X_train, y_train)
rf.fit(X_train, y_train)

y_pred_log = log.predict(X_test)
y_pred_knn = knn.predict(X_test)
y_pred_rf = rf.predict(X_test)

acc_log = accuracy_score(y_test, y_pred_log)
acc_knn = accuracy_score(y_test, y_pred_knn)
acc_rf = accuracy_score(y_test, y_pred_rf)

st.subheader("Acurácias (modelos base)")
st.write(f"Acurácia — Logistic Regression: {acc_log:.3f}")
st.write(f"Acurácia — KNN: {acc_knn:.3f}")
st.write(f"Acurácia — Random Forest: {acc_rf:.3f}")

# ---------------------------
# 6. Avaliação detalhada e comparação
# ---------------------------
st.subheader("Avaliação detalhada e comparação")
models = {'Logistic Regression': log, 'KNN': knn, 'Random Forest': rf}
results = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        try:
            probs = model.decision_function(X_test)
        except Exception:
            probs = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, probs)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    results[name] = {'accuracy': acc, 'roc_auc': auc, 'precision': prec, 'recall': rec, 'f1': f1}

    st.markdown(f"##### {name}")
    st.text(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Matriz de confusão — {name}')
    ax.set_xlabel('Predito')
    ax.set_ylabel('Real')
    st.pyplot(fig, clear_figure=True)

results_df = pd.DataFrame(results).T.sort_values('roc_auc', ascending=False)
st.dataframe(results_df.style.format({c: "{:.3f}" for c in results_df.columns}))

# Curvas ROC
st.subheader("Curvas ROC")
fig, ax = plt.subplots(figsize=(8, 6))
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        try:
            probs = model.decision_function(X_test)
        except Exception:
            probs = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], '--', color='grey')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves')
ax.legend()
st.pyplot(fig, clear_figure=True)

# ---------------------------
# 7. Importância de features (Random Forest)
# ---------------------------
st.subheader("Importância de features — Random Forest")
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=ax)
ax.set_title('Importância das features — Random Forest')
st.pyplot(fig, clear_figure=True)

# ---------------------------
# 8. Validação cruzada e GridSearch (como no notebook)
# ---------------------------
st.subheader("Validação cruzada (Random Forest)")
X_scaled_full = StandardScaler().fit_transform(X)
cv_scores = cross_val_score(rf, X_scaled_full, y, cv=5, scoring='roc_auc')
st.write("Random Forest CV AUC (5-fold):", float(cv_scores.mean()))

st.subheader("GridSearch (Random Forest)")
param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 5, 10]}
gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='roc_auc')
gs.fit(X_train, y_train)
st.write("Melhores parâmetros RF:", gs.best_params_)

best_rf = gs.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)
st.write("Acurácia do RF ajustado:", float(accuracy_score(y_test, y_pred_best_rf)))

# ---------------------------
# 9. Persistência do melhor modelo (igual ao original)
# ---------------------------
joblib.dump(best_rf, 'best_random_forest_model.joblib')
st.success("Modelo salvo: best_random_forest_model.joblib")
