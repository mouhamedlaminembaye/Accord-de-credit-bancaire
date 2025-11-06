import streamlit as st
import pandas as pd
import pickle

# Charger le pipeline complet
with open('pipeline_credit.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Prédiction Crédit Bancaire", page_icon="💳")
st.title("💳 Prédiction d'Octroi de Crédit Bancaire")
st.write("Remplissez les informations du client pour prédire si le crédit sera accordé.")

# --- Entrées utilisateur ---
st.header("🧍 Informations du client")

ApplicantIncome = st.number_input("Revenu du demandeur (€)", 0, 20000, 2500)
CoapplicantIncome = st.number_input("Revenu du co-demandeur (€)", 0, 20000, 0)
LoanAmount = st.number_input("Montant du crédit demandé (€)", 0, 500000, 100000)
Loan_Amount_Term = st.number_input("Durée du crédit (mois)", 1, 360, 360)
Credit_History = st.selectbox("Historique de crédit", [1, 0])
Dependents = st.selectbox("Nombre de personnes à charge", ['0','1','2','3+'])
Education = st.selectbox("Éducation", ["Graduate", "Not Graduate"])
Gender = st.selectbox("Genre", ["Male", "Female"])
Married = st.selectbox("Marié(e)", ["Yes", "No"])
Self_Employed = st.selectbox("Travailleur indépendant", ["Yes", "No"])

# --- Préparer DataFrame d'entrée ---
input_data = pd.DataFrame({
    'ApplicantIncome': [ApplicantIncome],
    'CoapplicantIncome': [CoapplicantIncome],
    'LoanAmount': [LoanAmount],
    'Loan_Amount_Term': [Loan_Amount_Term],
    'Credit_History': [Credit_History],
    'Dependents': [Dependents],
    'Education': [Education],
    'Gender': [Gender],
    'Married': [Married],
    'Self_Employed': [Self_Employed]
})

# --- Prédiction ---
if st.button("🔮 Prédire l'octroi du crédit"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    st.subheader("Résultat de la prédiction :")
    if prediction == 1:
        st.success(f"✅ Crédit **accordé** avec une probabilité de {proba*100:.2f}%")
    else:
        st.error(f"❌ Crédit **refusé** avec une probabilité de {proba*100:.2f}%")

    # Affichage graphique optionnel
    st.progress(float(proba))
