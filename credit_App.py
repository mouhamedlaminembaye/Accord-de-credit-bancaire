import streamlit as st
import pandas as pd
import pickle

# -----------------------
# Charger le pipeline
# -----------------------
with open('pipeline_credit.pkl', 'rb') as f:
    model = pickle.load(f)

# -----------------------
# Configuration de la page
# -----------------------
st.set_page_config(page_title="Prédiction Crédit Bancaire", page_icon="💳")
st.title("💳 Prédiction d'Octroi de Crédit Bancaire")
st.write("Remplissez les informations du client pour prédire si le crédit sera accordé.")

# -----------------------
# Entrées utilisateur
# -----------------------
st.header("🧍 Informations du client")

input_dict = {
    'ApplicantIncome': st.number_input("Revenu du demandeur (€)", min_value=0, max_value=20000, value=2500, step=100),
    'CoapplicantIncome': st.number_input("Revenu du co-demandeur (€)", min_value=0, max_value=20000, value=0, step=100),
    'LoanAmount': st.number_input("Montant du crédit demandé (€)", min_value=0, max_value=500000, value=100000, step=1000),
    'Loan_Amount_Term': st.number_input("Durée du crédit (mois)", min_value=1, max_value=360, value=360, step=12),
    'Credit_History': st.selectbox("Historique de crédit", [1, 0]),
    'Dependents': st.selectbox("Nombre de personnes à charge", ['0','1','2','3+']),
    'Education': st.selectbox("Éducation", ["Graduate", "Not Graduate"]),
    'Gender': st.selectbox("Genre", ["Male", "Female"]),
    'Married': st.selectbox("Marié(e)", ["Yes", "No"]),
    'Self_Employed': st.selectbox("Travailleur indépendant", ["Yes", "No"])
}

# Créer le DataFrame pour la prédiction
input_data = pd.DataFrame([input_dict])

# -----------------------
# Prédiction
# -----------------------
if st.button("🔮 Prédire le crédit"):
    # Prédiction du modèle
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    # Affichage du résultat
    st.subheader("Résultat de la prédiction :")
    if prediction == 1:
        st.success(f"✅ Crédit **accordé** avec une probabilité de {proba*100:.2f}%")
    else:
        st.error(f"❌ Crédit **refusé** avec une probabilité de {proba*100:.2f}%")

    # Barre de progression pour la probabilité
    st.progress(float(proba))
