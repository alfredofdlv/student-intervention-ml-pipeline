import streamlit as st
import pandas as pd
import joblib
st.set_page_config(
    page_title="🎓 Student Pass/Fail Predictor",
    page_icon="🎓",
    layout="wide"
)
# ─── Load the trained model ──────────────────────────────────────────────────
try:
    model = joblib.load("best_logreg_model.joblib")
    st.sidebar.success("✅ Model loaded")
except FileNotFoundError:
    st.sidebar.error("❌ Could not find 'best_logreg_model.joblib'.")
    st.stop()
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {e}")
    st.stop()



st.title("🎓 Student Pass/Fail Prediction App")
st.markdown(
    "Use the form below to enter a student's data and see whether they are likely to pass or fail."
)

# ─── Define inputs ───────────────────────────────────────────────────────────
with st.expander("👤 Demographics & Family Background", expanded=True):
    cols = st.columns(3)
    with cols[0]:
        st.radio("🏫 School", ["GP", "MS"], key="school")
        st.radio("🚻 Sex", ["F", "M"], key="sex")
        st.number_input("🎂 Age", min_value=10, max_value=25, value=17, key="age")
    with cols[1]:
        st.radio("🏠 Address", ["U", "R"], key="address")
        st.radio("👪 Family Size", ["LE3", "GT3"], key="famsize")
        st.radio("📝 Parent Status", ["T", "A"], key="Pstatus")
    with cols[2]:
        st.selectbox("👨‍👩‍👧 Guardian", ["father", "mother", "other"], key="guardian")
        st.selectbox("🎯 Reason for Enrollment", ["course", "home", "other", "reputation"], key="reason")
        st.number_input("📅 Absences", min_value=0, max_value=100, value=0, key="absences")

with st.expander("📚 Education & Support", expanded=False):
    cols = st.columns(4)
    ord_sliders = {
        "Mother's education (Medu)":  ("Medu", 0, 4, 2),
        "Father's education (Fedu)":  ("Fedu", 0, 4, 2),
        "Family relationship":        ("famrel", 1, 5, 3),
        "Free time":                  ("freetime", 1, 5, 3),
        "Going out":                  ("goout", 1, 5, 3),
        "Workday alcohol (Dalc)":     ("Dalc", 1, 5, 1),
        "Weekend alcohol (Walc)":     ("Walc", 1, 5, 2),
        "Health":                     ("health", 1, 5, 3),
        "Study time":                 ("studytime", 1, 4, 2),
        "Travel time":                ("traveltime", 1, 4, 1),
        "Past failures":              ("failures", 0, 3, 0),
    }
    i = 0
    for label, (key, mn, mx, df) in ord_sliders.items():
        with cols[i % 4]:
            st.slider(f"📊 {label}", mn, mx, df, key=key)
        i += 1

    cols2 = st.columns(4)
    binaries = [
        ("🎓 School support", "schoolsup"),
        ("🏠 Family support",  "famsup"),
        ("💰 Paid classes",    "paid"),
        ("🎨 Activities",     "activities"),
        ("🌱 Nursery",        "nursery"),
        ("🌐 Internet",       "internet"),
        ("❤️ Romantic",       "romantic"),
        ("🎓 Wants higher ed","higher"),
    ]
    for idx, (label, key) in enumerate(binaries):
        with cols2[idx % 4]:
            st.checkbox(label, key=key)

with st.expander("💼 Jobs & Lifestyle", expanded=False):
    cols = st.columns(3)
    job_opts  = ["at_home", "health", "other", "services", "teacher"]
    with cols[0]:
        st.selectbox("👩‍⚕️ Mother's Job", job_opts, key="Mjob")
        st.selectbox("👨‍⚕️ Father's Job", job_opts, key="Fjob")
    with cols[1]:
        # (Note: these are engineered features, leave as numeric inputs)
        st.number_input("⚖️ Leisure–study balance", key="leisure_balance", value=0.0, step=0.1)
        st.number_input("🗣️ Sociality index",         key="sociality_index",  value=0.0, step=0.1)
        st.number_input("🍻 Alcohol index",            key="alcohol_index",     value=0.0, step=0.1)
    with cols[2]:
        st.number_input("⚕️ Health risk score",      key="health_risk_score", value=0.0, step=0.1)
        st.number_input("💻 Tech access",             key="tech_access",        value=0.0, step=0.1)
        st.number_input("🆘 Support mismatch",        key="support_mismatch",   value=0.0, step=0.1)

    cols3 = st.columns(2)
    with cols3[0]:
        st.number_input("➕ Support sum",        key="support_sum",      value=0.0, step=0.1)
        st.number_input("📉 Absence ratio",     key="absence_ratio",    value=0.0, step=0.01)
        st.checkbox("🚌 Long commuter flag",   key="long_commuter_flag")
    with cols3[1]:
        st.number_input("👵 Guardian father",   key="guardian_father", value=0.0, step=0.1)
        st.number_input("👩 Guardian mother",   key="guardian_mother", value=0.0, step=0.1)
        st.number_input("👤 Guardian other",    key="guardian_other",  value=0.0, step=0.1)

# ─── Build the DataFrame ────────────────────────────────────────────────────
FEATURE_NAMES = [
    'Mjob_at_home','Mjob_health','Mjob_other','Mjob_services','Mjob_teacher',
    'Fjob_at_home','Fjob_health','Fjob_other','Fjob_services','Fjob_teacher',
    'guardian_father','guardian_mother','guardian_other',
    'reason_course','reason_home','reason_other','reason_reputation',
    'Medu','Fedu','famrel','freetime','goout',
    'Dalc','Walc','health','studytime','traveltime','failures',
    'school','sex','address','famsize','Pstatus',
    'schoolsup','famsup','paid','activities','nursery',
    'internet','romantic','higher','age','absences',
    'leisure_balance','sociality_index','alcohol_index','health_risk_score',
    'tech_access','support_sum','support_mismatch','absence_ratio','long_commuter_flag'
]

if st.button("🚀 Predict Outcome"):
    # collect inputs in the order FEATURE_NAMES expects
    vals = {}
    for f in FEATURE_NAMES:
        vals[f] = st.session_state.get(f, 0.0)
    input_df = pd.DataFrame([vals], columns=FEATURE_NAMES)

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0, 1]

    if pred == 0:
        st.success("🎉 Predicted: PASS")
    else:
        st.error("⚠️ Predicted: FAIL")

    st.write(f"🔢 Probability of failing: **{prob:.2f}**")
