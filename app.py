import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load("best_model.pkl")

st.set_page_config(
    page_title="💼 Salary Predictor | AI-Powered",
    page_icon="💼",
    layout="wide"
)

st.title("💸Explore Smarter Income Insights with AI Assistance 💼")

# Sidebar 
with st.sidebar:
    st.markdown("### 📝 Employee Input Form")
    st.markdown("Fill in the details below to predict salary class")

    age = st.slider("🎂 Age", 18, 70, 30)
    gender = st.selectbox("🧑‍🤝‍🧑 Gender", ["Male", "Female", "Other"])
    education = st.selectbox("🎓 Education Level", ["Bachelors", "Masters", "Doctorate", "High school", "Some-college", "Other"])
    workclass = st.selectbox("🏢 Work Class", ["Private", "Self-emp", "Government"])
    occupation = st.selectbox("💼 Occupation", ["Software Engineer", "Business Analyst", "Video Editor","Tech-support", "Sales", "Executive", "Clerical", "Other"])
    experience = st.slider("📆 Years of Experience", 0, 50, 1)
    hours_per_week = st.slider("⏱️ Hours per Week", 1, 99, 40)

    st.markdown("### 🏦 Personal & Financial Information")
    marital_status = st.selectbox("💍 Marital Status", ["Never-married", "Married-civ-spouse", "Divorced"])
    relationship = st.selectbox("👪 Relationship", ["Husband", "Wife", "Not-in-family", "Unmarried"])
    native_country = st.selectbox("🌐 Native Country", [ "India","United-States" "Mexico", "Germany", "Other"])
    final_weight = st.number_input("⚖️ Final Weight", value=100000)
    capital_gain = st.number_input("📈 Capital Gain", value=0)
    capital_loss = st.number_input("📉 Capital Loss", value=0)

# Main Container
st.markdown("<h1 style='text-align:center;'>💼 AI-Powered Salary Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Predict whether an employee earns more or less than 50K/month</p>", unsafe_allow_html=True)

st.divider()

#  Prediction Section 
st.markdown("## 🎯 Predict Individual Salary")
with st.expander("🔍 Review Input & Predict", expanded=True):
    st.markdown("### 🧾 Your Selections")
    st.write(f"**Age :** {age}")
    st.write(f"**Gender:** {gender}")
    st.write(f"**Education:** {education}")
    st.write(f"**Work Class:** {workclass}")
    st.write(f"**Occupation:** {occupation}")
    st.write(f"**Experience:** {experience}")
    st.write(f"**Hours per Week:** {hours_per_week}")
    st.write(f"**Marital Status:** {marital_status}")
    st.write(f"**Relationship:** {relationship}")
    st.write(f"**Native Country:** {native_country}")
    st.write(f"**Final Weight:** {final_weight}")
    st.write(f"**Capital Gain:** {capital_gain}")
    st.write(f"**Capital Loss:** {capital_loss}")

if st.button("🚀 Predict Salary Class"):
    # Create DataFrame from inputs
    input_dict = {
        'age': age,
        'gender': gender,
        'education': education,
        'workclass': workclass,
        'occupation': occupation,
        'experience': experience,
        'hours_per_week': hours_per_week,
        'marital_status': marital_status,
        'relationship': relationship,
        'native_country': native_country,
        'final_weight': final_weight,
        'capital_gain': capital_gain,
        'capital_loss': capital_loss
    }

    input_df = pd.DataFrame([input_dict])

    # Manual encoding (you must match this with training preprocessing)
    input_df['gender'] = input_df['gender'].map({'Male': 1, 'Female': 0, 'Other': 2})
    input_df['education'] = input_df['education'].map({'Bachelors': 13, 'Masters': 1, 'Doctorate': 16, 'High school': 9, 'Some-college': 10, 'Other': 15})
    input_df['workclass'] = input_df['workclass'].map({'Private': 4, 'Self-emp': 2, 'Government': 6})
    input_df['occupation'] = input_df['occupation'].map({
        'Software Engineer': 10, 'Business Analyst': 9, 'Video Editor': 0,
        'Tech-support': 13, 'Sales': 4, 'Executive': 7, 'Clerical': 1, 'Other': 8
    })
    input_df['marital_status'] = input_df['marital_status'].map({'Never-married': 4, 'Married-civ-spouse': 2, 'Divorced': 0})
    input_df['relationship'] = input_df['relationship'].map({'Husband': 0, 'Wife': 5, 'Not-in-family': 1, 'Unmarried': 4})
    input_df['native_country'] = input_df['native_country'].map({
        'India': 0, 'United-States': 39
        , 'Mexico': 2, 'Germany': 3, 'Other': 4
    })

    input_df = input_df.fillna(0)

    # Prediction
    prediction = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    confidence = round(max(probs) * 100, 2)

    st.success(f"💰 Predicted Salary Class: **{prediction}**")
    st.info(f"📊 Prediction Confidence: **{confidence}%**")

    # Bar chart
    st.bar_chart(pd.DataFrame(probs, index=model.classes_, columns=["Probability"]))

st.divider()

# Metrics Section
st.markdown("## 📊 Model Performance Metrics")

metric_style = """
<div style='background-color:{bg}; padding:20px; border-radius:10px; text-align:center; color:white;'>
    <h4>{label}</h4>
    <h2>{value}</h2>
</div>
"""

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(metric_style.format(label="🎯 Accuracy", value="87.7%", bg="#4CAF50"), unsafe_allow_html=True)
with col2:
    st.markdown(metric_style.format(label="📌 Precision", value="87%", bg="#2196F3"), unsafe_allow_html=True)
with col3:
    st.markdown(metric_style.format(label="🔁 Recall", value="87%", bg="#FF9800"), unsafe_allow_html=True)
with col4:
    st.markdown(metric_style.format(label="🧮 F1 Score", value="86%", bg="#9C27B0"), unsafe_allow_html=True)

st.divider()

# Batch Prediction 
st.markdown("## 🗃️ Batch Prediction")
st.write("📂 Upload a CSV to predict salaries for multiple entries:")

uploaded_file = st.file_uploader("📎 Choose CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📄 Uploaded Data Preview:")
    st.dataframe(df.head())

    st.button("⚙️ Run Batch Prediction")  


# Footer 
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>"
    "🔧 Developed by <b> Srijan Roy </b> | IEM, CSE KOLKATA"
    "<br>Built using <a href='https://streamlit.io' target='_blank'>Streamlit</a> 🚀"
    "</div>",
    unsafe_allow_html=True
)
