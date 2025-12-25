import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gc
import plotly.express as px

# ==================================
# 1. Page Config & Professional Custom CSS
# ==================================
st.set_page_config(page_title="Diabetes AI Lab", page_icon="🏥", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #007bff; color: white; font-weight: bold; }
    .prediction-card { padding: 20px; border-radius: 15px; border-left: 5px solid #007bff; background-color: white; }
    </style>
    """, unsafe_allow_html=True)

# ==================================
# 2. Advanced Language Management
# ==================================
if 'lang' not in st.session_state:
    st.session_state.lang = 'English'

# شريط علوي أنيق لتغيير اللغة
col_l1, col_l2 = st.columns([8, 2])
with col_l2:
    lang_toggle = st.segmented_control("Language / اللغة", ["English", "العربية"], default=st.session_state.lang)
    if lang_toggle: st.session_state.lang = lang_toggle

texts = {
    "English": {
        "title": "🏥 Diabetes Intelligence System",
        "subtitle": "Advanced Diagnostic Analysis & Predictive Modeling",
        "tab_predict": "🤖 Smart Prediction",
        "tab_viz": "📊 Data Insights",
        "btn": "Run AI Diagnosis",
        "res": "Diagnostic Report",
        "prob": "Diabetes Probability",
        "cat_header": "📁 Patient Background",
        "num_header": "📉 Vital Biomarkers",
        "viz_title": "Dataset Exploration"
    },
    "العربية": {
        "title": "🏥 نظام ذكاء السكري العالمي",
        "subtitle": "التحليل التشخيصي المتقدم ونمذجة التوقعات",
        "tab_predict": "🤖 التنبؤ الذكي",
        "tab_viz": "📊 رؤى البيانات",
        "btn": "تشغيل التشخيص الذكي",
        "res": "التقرير التشخيصي النهائي",
        "prob": "احتمالية الإصابة بالسكري",
        "cat_header": "📁 المعلومات الشخصية",
        "num_header": "📉 القياسات الحيوية",
        "viz_title": "استكشاف وتوزيع البيانات"
    }
}
L = texts[st.session_state.lang]


# ==================================
# 3. Assets Loading (Cached)
# ==================================
@st.cache_resource
def load_resources():
    data = pd.read_csv(
        r"C:\Users\USER\AI-Projects\ML Projects\Diabetes Prediction\Original Data\Dataset\diabetes_dataset.csv",
        nrows=10000)
    data_final_cols = pd.read_csv(
        r"C:\Users\USER\AI-Projects\ML Projects\Diabetes Prediction\Original Data\Dataset\diabetes_final.csv",
        nrows=0).columns.tolist()
    if 'Unnamed: 0' in data_final_cols: data_final_cols.remove('Unnamed: 0')

    model = joblib.load(
        r"C:\Users\USER\AI-Projects\ML Projects\Diabetes Prediction\Original Data\Supervised ML\XGB_model_diabetes_OriginalData.pkl")
    scaler = joblib.load(
        r"C:\Users\USER\AI-Projects\ML Projects\Diabetes Prediction\Original Data\Preprocessing\scaler_diabetes.pkl")
    OHE = joblib.load(
        r"C:\Users\USER\AI-Projects\ML Projects\Diabetes Prediction\Original Data\Preprocessing\OneHotEncoder_diabetes.pkl")
    OE = joblib.load(
        r"C:\Users\USER\AI-Projects\ML Projects\Diabetes Prediction\Original Data\Preprocessing\OrdinalEncoder_diabetes.pkl")
    return data, data_final_cols, model, scaler, OHE, OE


data, final_cols_list, model, scaler, OHE, OE = load_resources()

# ==================================
# 4. Interface Structure (Tabs)
# ==================================
t1, t2 = st.tabs([L["tab_predict"], L["tab_viz"]])

with t1:
    st.title(L["title"])
    st.markdown(f"*{L['subtitle']}*")

    # تنظيم المدخلات في حاويات ملونة
    with st.container():
        st.subheader(L["cat_header"])
        cat_cols = data.select_dtypes(include=['object']).columns.tolist()
        user_inputs = {}
        c1, c2, c3 = st.columns(3)
        for i, col in enumerate(cat_cols):
            with [c1, c2, c3][i % 3]:
                user_inputs[col] = st.selectbox(f"📍 {col.replace('_', ' ').title()}", data[col].unique())

    st.divider()

    with st.container():
        st.subheader(L["num_header"])
        num_cols = data.select_dtypes(include=['number']).columns.tolist()
        # استبعاد الـ target من المدخلات لو وجد
        if 'diagnosed_diabetes' in num_cols: num_cols.remove('diagnosed_diabetes')

        c1, c2, c3 = st.columns(3)
        for i, col in enumerate(num_cols):
            with [c1, c2, c3][i % 3]:
                user_inputs[col] = st.slider(f"🔢 {col.replace('_', ' ').title()}",
                                             float(data[col].min()), float(data[col].max()), float(data[col].mean()))

    # زر التشخيص بشكل مميز
    st.write("---")
    if st.button(L["btn"]):
        # --- Preprocessing (كودك الأصلي كما هو) ---
        df_input = pd.DataFrame([user_inputs])

        ord_features = ['education_level', 'income_level']
        df_input[ord_features] = OE.transform(df_input[ord_features])

        ohe_required_cols = OHE.feature_names_in_.tolist()
        X_nom = df_input[ohe_required_cols]
        nom_encoded = OHE.transform(X_nom)
        nom_df = pd.DataFrame(nom_encoded, columns=OHE.get_feature_names_out())

        remaining_cols = [c for c in df_input.columns if c not in ohe_required_cols]
        processed_df = pd.concat([df_input[remaining_cols], nom_df], axis=1)

        num_features_in_scaler = scaler.feature_names_in_.tolist()
        processed_df[num_features_in_scaler] = scaler.transform(processed_df[num_features_in_scaler])

        f_cols = [c for c in final_cols_list if c != 'diagnosed_diabetes']
        processed_df = processed_df[f_cols]

        prob = model.predict_proba(processed_df)[0][1]
        prob_float = float(prob)

        # --- عرض النتيجة بشكل احترافي ---
        st.balloons() if prob_float < 0.3 else None

        st.markdown(f"### {L['res']}")
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.metric(L["prob"], f"{prob_float * 100:.1f}%")

        with res_col2:
            if prob_float > 0.5:
                st.error("🚨 " + (
                    "High Risk: Medical Consultation Required" if st.session_state.lang == 'English' else "خطر مرتفع: استشارة طبية مطلوبة"))
            else:
                st.success("✨ " + (
                    "Low Risk: Continue your healthy lifestyle" if st.session_state.lang == 'English' else "خطر منخفض: حافظ على نمط حياتك الصحي"))

        st.progress(prob_float)
        gc.collect()

with t2:
    st.title(L["viz_title"])
    st.write("Explore relationships between features in the dataset")

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        v_feat = st.selectbox("Select Feature to Analyze", num_cols)
    with col_v2:
        v_type = st.radio("Chart Type", ["Box Plot", "Distribution"], horizontal=True)

    if v_type == "Box Plot":
        fig = px.box(data, x="diagnosed_diabetes", y=v_feat, color="diagnosed_diabetes",
                     title=f"{v_feat} vs Diabetes Status", template="plotly_white")
    else:
        fig = px.histogram(data, x=v_feat, color="diagnosed_diabetes", marginal="rug",
                           title=f"Distribution of {v_feat}", template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)