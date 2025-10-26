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

# Informações do projeto (original do notebook)
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

# Carregamento do dataset (mantido)
CSV_PATH = "heart_failure_clinical_records_dataset.csv"
try:
    df = pd.read_csv(CSV_PATH)
except Exception as e:
    st.error(f"Não foi possível carregar o CSV '{CSV_PATH}'. Erro: {e}")
    st.stop()

# Abas para organização da UI (EDA, Modelagem, Avaliação)
tab_eda, tab_model, tab_eval = st.tabs(["EDA", "Modelagem", "Avaliação"])  # organização da navegação em seções

with tab_eda:
    st.subheader("Amostra do dataset")
    st.dataframe(df.head(), use_container_width=True)

    buf = io.StringIO()
    df.info(buf=buf)
    st.subheader("Info do DataFrame")
    st.code(buf.getvalue(), language="text")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("Estatísticas descritivas")
        st.dataframe(df.describe().T, use_container_width=True)
    with c2:
        st.subheader("Valores nulos por coluna")
        st.dataframe(df.isnull().sum().to_frame("nulls"), use_container_width=True)

    st.divider()
    c3, c4 = st.columns([1, 1], gap="large")
    with c3:
        st.subheader("Distribuição do alvo")
        fig, ax = plt.subplots(figsize=(4.5, 3.2))  # menor
        sns.countplot(x='DEATH_EVENT', data=df, ax=ax)
        ax.set_title('DEATH_EVENT (0=vivo, 1=óbito)')
        st.pyplot(fig, clear_figure=True)
    with c4:
        st.subheader("Mapa de correlação")
        # matriz de correlação com todas as colunas numéricas, como no notebook
        corr = df.corr(numeric_only=True)  # idem ao comportamento clássico do pandas
    
        # figura ampla, matriz completa, anotada e com 'coolwarm' (sem máscara)
        fig, ax = plt.subplots(figsize=(12, 10))  # tamanho amplo como no script original
        sns.heatmap(
            corr,
            annot=True, fmt=".2f",
            cmap="coolwarm",
            ax=ax
        )
        ax.set_title("Mapa de Correlação")
        st.pyplot(fig, clear_figure=True)



with tab_model:
    if 'DEATH_EVENT' not in df.columns:
        st.error("Coluna 'DEATH_EVENT' não encontrada no dataset.")
        st.stop()

    X = df.drop('DEATH_EVENT', axis=1)
    y = df['DEATH_EVENT']

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    st.subheader("Divisão e normalização")
    st.caption(f"Treino: {X_train.shape} | Teste: {X_test.shape}")

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
    k1, k2, k3 = st.columns(3)
    k1.metric("Logistic Regression", f"{acc_log:.3f}")
    k2.metric("KNN (k=5)", f"{acc_knn:.3f}")
    k3.metric("Random Forest", f"{acc_rf:.3f}")

with tab_eval:
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
        st.code(classification_report(y_test, y_pred, zero_division=0), language="text")

        col_cm, col_empty = st.columns([1, 1])
        with col_cm:
            fig, ax = plt.subplots(figsize=(4.8, 3.6))  # menor
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'Matriz de confusão — {name}')
            ax.set_xlabel('Predito')
            ax.set_ylabel('Real')
            st.pyplot(fig, clear_figure=True)

    results_df = pd.DataFrame(results).T.sort_values('roc_auc', ascending=False)
    st.dataframe(results_df.style.format({c: "{:.3f}" for c in results_df.columns}), use_container_width=True)

    st.subheader("Curvas ROC")
    # ROC menor e em duas colunas para organização
    cols = st.columns(2, gap="large")
    model_list = list(models.items())
    for idx, (name, model) in enumerate(model_list):
        with cols[idx % 2]:
            fig, ax = plt.subplots(figsize=(5.0, 3.4))  # menor
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)[:, 1]
            else:
                try:
                    probs = model.decision_function(X_test)
                except Exception:
                    probs = model.predict(X_test)
            fpr, tpr, _ = roc_curve(y_test, probs)
            auc = roc_auc_score(y_test, probs)
            ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
            ax.plot([0, 1], [0, 1], '--', color='grey')
            ax.set_xlabel('FPR')
            ax.set_ylabel('TPR')
            ax.set_title(f'ROC — {name}')
            ax.legend(frameon=False, fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
            
    st.divider()
    st.subheader("Validação cruzada e GridSearch")
    X_scaled_full = StandardScaler().fit_transform(df.drop('DEATH_EVENT', axis=1))
    cv_scores = cross_val_score(rf, X_scaled_full, y, cv=5, scoring='roc_auc')
    st.caption(f"Random Forest CV AUC (5-fold): {float(cv_scores.mean()):.3f}")

    param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 5, 10]}
    gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='roc_auc')
    gs.fit(X_train, y_train)
    st.caption(f"Melhores parâmetros RF: {gs.best_params_}")

    best_rf = gs.best_estimator_
    y_pred_best_rf = best_rf.predict(X_test)
    st.caption(f"Acurácia do RF ajustado: {float(accuracy_score(y_test, y_pred_best_rf)):.3f}")

    joblib.dump(best_rf, 'best_random_forest_model.joblib')
    st.success("Modelo salvo: best_random_forest_model.joblib")
