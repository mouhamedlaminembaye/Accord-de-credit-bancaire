import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prédiction de crédit 💳", page_icon="💰", layout="centered")

st.title("Application de prédiction d'accord de crédit")
st.write("Entrez les informations du client pour obtenir une prédiction automatique.")

@st.cache_resource
def load_model():
    return joblib.load("model_credit.joblib")

model = load_model()

with st.form("credit_form"):
    st.subheader("Informations sur le client")

    gender = st.selectbox("Genre", ["Male", "Female"])
    married = st.selectbox("Marié(e)", ["Yes", "No"])
    dependents = st.selectbox("Nombre de personnes à charge", ["0", "1", "2", "3+"])
    education = st.selectbox("Niveau d’éducation", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Travail indépendant", ["Yes", "No"])
    property_area = st.selectbox("Zone du bien immobilier", ["Urban", "Semiurban", "Rural"])

    applicant_income = st.number_input("Revenu du demandeur", min_value=0)
    coapplicant_income = st.number_input("Revenu du co-demandeur", min_value=0)
    loan_amount = st.number_input("Montant du prêt (en milliers)", min_value=0)
    loan_amount_term = st.number_input("Durée du prêt (en mois)", min_value=0)
    credit_history = st.selectbox("Historique de crédit", [1.0, 0.0])

    submitted = st.form_submit_button("🔮 Prédire")

if submitted:
    # 🔹 Encodage manuel des variables catégorielles
    gender_map = {"Male": 1, "Female": 0}
    married_map = {"Yes": 1, "No": 0}
    education_map = {"Graduate": 1, "Not Graduate": 0}
    self_employed_map = {"Yes": 1, "No": 0}
    property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
    dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3}

    input_data = pd.DataFrame({
        'Gender': [gender_map[gender]],
        'Married': [married_map[married]],
        'Dependents': [dependents_map[dependents]],
        'Education': [education_map[education]],
        'Self_Employed': [self_employed_map[self_employed]],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_map[property_area]]
    })

    try:
        prediction = model.predict(input_data)[0]
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_data)[0, 1]
        else:
            prob = None

        st.subheader("Résultat de la prédiction")
        if prediction == 1:
            st.success(f"✅ Crédit ACCORDÉ (probabilité: {prob:.2%})" if prob else "✅ Crédit ACCORDÉ !")
        else:
            st.error(f"❌ Crédit REFUSÉ (probabilité: {prob:.2%})" if prob else "❌ Crédit REFUSÉ.")
    except Exception as e:
        st.error(f"Erreur pendant la prédiction : {e}")
