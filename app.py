import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
import io

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

st.set_page_config(page_title="Previsão Insuficiência Cardíaca", layout="wide")

st.title("Previsão de Óbito por Insuficiência Cardíaca")
st.markdown("Projeto completo com análise, modelagem e avaliação de algoritmos ML.")

with st.sidebar:
    st.header("Sobre o Dataset")
    st.markdown("""
    - 299 registros clínicos  
    - 12 variáveis clínicas + alvo  
    - Objetivo: prever óbito (`DEATH_EVENT`)  
    """)
    if st.button("Mostrar informações do dataframe"):
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
st.subheader("Visualização dos Dados")
st.write(f"Dimensão do dataset: {df.shape[0]} registros e {df.shape[1]} colunas")
st.dataframe(df.head(), height=250)

st.subheader("Análise Exploratória")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Distribuição de DEATH_EVENT**")
    fig1, ax1 = plt.subplots(figsize=(6,4))
    sns.countplot(x='DEATH_EVENT', data=df, ax=ax1)
    ax1.set_title('Distribuição de DEATH_EVENT (0 = vivo, 1 = óbito)')
    st.pyplot(fig1)

    st.markdown("**Valores Nulos por Coluna**")
    st.write(df.isnull().sum())

with col2:
    st.markdown("**Mapa de Correlação**")
    fig2, ax2 = plt.subplots(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax2)
    ax2.set_title('Correlação entre Variáveis')
    st.pyplot(fig2)

st.subheader("Pré-processamento")
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

st.write(f'Conjunto de Features: {X.shape}')
st.write(f'Conjunto Target: {y.shape}')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

st.write(f'Dados de treino: {X_train.shape} | Dados de teste: {X_test.shape}')

st.subheader("Treinamento e Resultados")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {'accuracy': acc}

    with st.expander(f"{name} - Relatório de Classificação"):
        st.text(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax_cm)
        ax_cm.set_title(f'Matriz de Confusão - {name}')
        ax_cm.set_xlabel('Predito')
        ax_cm.set_ylabel('Real')
        st.pyplot(fig_cm)

acc_df = pd.DataFrame(results).T
st.write("Acurácia dos Modelos:")
st.dataframe(acc_df)

st.subheader("Curvas ROC")
fig_roc, ax_roc = plt.subplots(figsize=(7,5))
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:,1]
    else:
        try:
            probs = model.decision_function(X_test)
        except:
            probs = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
ax_roc.plot([0,1], [0,1], 'k--')
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curves")
ax_roc.legend()
st.pyplot(fig_roc)

st.subheader("Importância das Features - Random Forest")
feat_imp = pd.Series(models['Random Forest'].feature_importances_, index=X.columns)
feat_imp = feat_imp.sort_values(ascending=False)

fig_feat, ax_feat = plt.subplots(figsize=(6,4))
sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=ax_feat)
ax_feat.set_title("Top Features por importância (Random Forest)")
st.pyplot(fig_feat)

st.subheader("Validação Cruzada e Otimização")

rf = models['Random Forest']
cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='roc_auc')
st.write(f"Random Forest CV AUC (5-fold): {cv_scores.mean():.3f}")

param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 5, 10]}
gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='roc_auc')
gs.fit(X_train, y_train)
st.write(f"Melhores parâmetros RF: {gs.best_params_}")

best_rf = gs.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)
st.write(f"Acurácia do RF ajustado: {accuracy_score(y_test, y_pred_best_rf):.3f}")

import joblib
joblib.dump(best_rf, 'best_random_forest_model.joblib')
st.write("Modelo salvo: best_random_forest_model.joblib")
