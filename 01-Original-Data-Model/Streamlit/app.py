import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gc

# ==================================
# 1. Page Config & Language
# ==================================
st.set_page_config(page_title="Diabetes Health Analytics", page_icon="🏥", layout="wide")

if 'lang' not in st.session_state:
    st.session_state.lang = 'English'

with st.sidebar:
    st.session_state.lang = st.selectbox("Language / اللغة", ["English", "العربية"])

texts = {
    "English": {"title": "🏥 Diabetes Prediction System", "btn": "Check Results", "res": "Prediction Results"},
    "العربية": {"title": "🏥 نظام التنبؤ بالسكري", "btn": "فحص النتائج", "res": "نتائج التوقع"}
}
L = texts[st.session_state.lang]


# ==================================
# 2. Load Assets
# ==================================

data = pd.read_csv(r"C:\Users\USER\AI-Projects\ML Projects\Diabetes Prediction\Original Data\Dataset\diabetes_dataset.csv", nrows=10000)
data_final = pd.read_csv(r"C:\Users\USER\AI-Projects\ML Projects\Diabetes Prediction\Original Data\Dataset\diabetes_final.csv", nrows=5)
data_final.drop('Unnamed: 0', axis=1, inplace=True)
model = joblib.load(r"C:\Users\USER\AI-Projects\ML Projects\Diabetes Prediction\Original Data\Supervised ML\XGB_model_diabetes_OriginalData.pkl")

scaler = joblib.load(r"C:\Users\USER\AI-Projects\ML Projects\Diabetes Prediction\Original Data\Preprocessing\scaler_diabetes.pkl")
OHE = joblib.load(r"C:\Users\USER\AI-Projects\ML Projects\Diabetes Prediction\Original Data\Preprocessing\OneHotEncoder_diabetes.pkl")
OE = joblib.load(r"C:\Users\USER\AI-Projects\ML Projects\Diabetes Prediction\Original Data\Preprocessing\OrdinalEncoder_diabetes.pkl")

# ==================================
# 3. Main Logic
# ==================================
st.title(L["title"])


num_cols = data.select_dtypes(include=['number']).columns.tolist()

cat_cols = data.select_dtypes(include=['object']).columns.tolist()

user_inputs = {}

col1, col2, col3 = st.columns(3)


for i, col in enumerate(cat_cols):
    current_col = [col1, col2, col3][i % 3]
    with current_col:
        options = data[col].unique().tolist()
        user_inputs[col] = st.selectbox(f"Select {col.replace('_', ' ').title()}", options)


for i, col in enumerate(num_cols):
    current_col = [col1, col2, col3][i % 3]
    with current_col:
        min_val = float(data[col].min())
        max_val = float(data[col].max())
        mean_val = float(data[col].mean())
        user_inputs[col] = st.slider(f"{col.replace('_', ' ').title()}", min_val, max_val, mean_val)

# ==================================
# 4. Prediction Button & Preprocessing
# ==================================


if st.button(L["btn"]):
    # 1. تحويل المدخلات لـ DataFrame
    df_input = pd.DataFrame([user_inputs])

    # 2. Ordinal Encoding (تأكد إن الترتيب هنا مطابق للي عملته وقت التدريب)
    ord_features = ['education_level', 'income_level']
    df_input[ord_features] = OE.transform(df_input[ord_features])

    # ---------------------------------------------------------
    # الحل السحري لمشكلة الـ OneHotEncoder
    # ---------------------------------------------------------
    # استخراج ترتيب الأعمدة اللي الـ OHE اتدرب عليها بالظبط
    ohe_required_cols = OHE.feature_names_in_.tolist()

    # اختيار الأعمدة من مدخلات اليوزر وترتيبها بنفس ترتيب الـ Encoder
    X_nom = df_input[ohe_required_cols]

    # دلوقتي الـ transform هيشتغل بدون أي Error
    nom_encoded = OHE.transform(X_nom)
    nom_df = pd.DataFrame(nom_encoded, columns=OHE.get_feature_names_out())
    # ---------------------------------------------------------

    # 3. الدمج النهائي (باقي الأعمدة + الـ OneHot)
    # بنشيل الأعمدة اللي دخلت في الـ OHE عشان ميتكرروش
    remaining_cols = [c for c in df_input.columns if c not in ohe_required_cols]
    processed_df = pd.concat([df_input[remaining_cols], nom_df], axis=1)

    # 4. الـ Scaling
    # تأكد إن scaler.feature_names_in_ موجودة عشان ترتب الأعمدة الرقمية برضه لو حصل خطأ
    num_features_in_scaler = scaler.feature_names_in_.tolist()
    processed_df[num_features_in_scaler] = scaler.transform(processed_df[num_features_in_scaler])

    # 5. الترتيب النهائي للموديل (XGBoost)
    final_cols = [c for c in data_final.columns.tolist() if c != 'diagnosed_diabetes']
    processed_df = processed_df[final_cols]

    # 6. التوقع

    prob = model.predict_proba(processed_df)[0][1]


    st.divider()
    st.header(L["res"])
    c_res1, c_res2 = st.columns(2)
    c_res1.metric("Probability / الاحتمالية", f"{prob * 100:.2f}%")

    if prob > 0.5:
        st.error("⚠️ High Risk Detected / احتمال إصابة مرتفع")
    else:
        st.success("✅ Healthy / احتمال إصابة منخفض")

