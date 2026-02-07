import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CreditGuard AI | EDA", page_icon="📊", layout="wide")

# --- LOAD MODEL ---
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('credit_model.pkl')
        # Load the dataset we saved in Step 1
        data = pd.read_csv('credit_data.csv')
        return model, data
    except FileNotFoundError:
        st.error("⚠️ Files missing! Run 'train_model.py' first.")
        st.stop()

model, df = load_resources()

st.title("🛡️ CreditGuard AI System")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["👤 Predict", "📈 EDA Dashboard", "📂 Batch Upload"])

# ==========================================
# TAB 1: PREDICTION (Standard)
# ==========================================
with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Applicant Details")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 80, 30)
        married = st.selectbox("Married?", ["Yes", "No"])
        bank_customer = st.selectbox("Existing Bank Customer?", ["Yes", "No"])
        years_employed = st.number_input("Years Employed", 0.0, 50.0, 2.5)
        prior_default = st.selectbox("Prior Default History?", ["Yes", "No"])
        employed = st.selectbox("Currently Employed?", ["Yes", "No"])
        credit_score = st.slider("Credit Score (0-20)", 0, 20, 5)
        income = st.number_input("Annual Income ($)", 0, 1000000, 50000)

    with col2:
        st.subheader("Result")
        if st.button("Predict"):
            # Prepare Input
            input_df = pd.DataFrame([{
                'Gender': 1 if gender == "Male" else 0,
                'Age': age,
                'Married': 1 if married == "Yes" else 0,
                'BankCustomer': 1 if bank_customer == "Yes" else 0,
                'YearsEmployed': years_employed,
                'PriorDefault': 1 if prior_default == "Yes" else 0,
                'Employed': 1 if employed == "Yes" else 0,
                'CreditScore': credit_score,
                'Income': income
            }])
            
            prob = model.predict_proba(input_df)[0][1]
            st.metric("Approval Probability", f"{prob:.1%}")
            st.progress(prob)

# ==========================================
# TAB 2: EDA DASHBOARD (New Feature!)
# ==========================================
with tab2:
    st.header("📊 Exploratory Data Analysis")
    st.write("Visualize the underlying patterns in the training dataset.")

    # 1. TOP METRICS
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Applicants", len(df))
    m2.metric("Approval Rate", f"{(df['Approved'].mean() * 100):.1f}%")
    m3.metric("Avg Income", f"${df['Income'].mean():,.0f}")

    st.divider()

    # 2. CHARTS ROW 1
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Approval Distribution")
        # Pie Chart
        fig1, ax1 = plt.subplots()
        df['Approved'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax1, colors=['#ff9999','#66b3ff'])
        ax1.set_ylabel('')
        st.pyplot(fig1)

    with c2:
        st.subheader("Income vs. Approval")
        # Box Plot
        fig2, ax2 = plt.subplots()
        sns.boxplot(x='Approved', y='Income', data=df, ax=ax2, palette="Set2")
        st.pyplot(fig2)

    st.divider()

    # 3. CHARTS ROW 2
    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Credit Score Impact")
        # Histogram
        fig3, ax3 = plt.subplots()
        sns.histplot(data=df, x='CreditScore', hue='Approved', multiple="stack", ax=ax3)
        st.pyplot(fig3)

    with c4:
        st.subheader("Correlation Matrix")
        # Heatmap
        # We need to convert text to numbers temporarily for the heatmap
        numeric_df = df.copy()
        for col in numeric_df.select_dtypes(include='object').columns:
            numeric_df[col] = numeric_df[col].astype('category').cat.codes
            
        fig4, ax4 = plt.subplots()
        sns.heatmap(numeric_df.corr(), cmap='coolwarm', ax=ax4)
        st.pyplot(fig4)

    # 4. RAW DATA
    with st.expander("🔎 View Raw Training Data"):
        st.dataframe(df.head(100))

# ==========================================
# TAB 3: BATCH UPLOAD (Kept simple)
# ==========================================
with tab3:
    st.header("📂 Bulk Processing")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Data Loaded. Implement prediction logic here.")