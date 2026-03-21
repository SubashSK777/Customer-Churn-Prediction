import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

# Page Config
st.set_page_config(page_title="Churn Crusher Dashboard", layout="wide")

# Custom CSS
st.markdown("""
<style>
.main { background-color: #0d1117; color: white; }
.stButton>button {
    width: 100%; height: 3.5rem;
    background-color: #1D9E75; color: white;
    font-size: 1.1rem; border-radius: 8px;
    border: none; box-shadow: 0 4px 10px rgba(29, 158, 117, 0.3);
}
.stButton>button:hover {
    background-color: #167d5c; transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div align="center">
  <h1 style="color: #1D9E75; font-size: 4em; font-family: 'Segoe UI', sans-serif;">💰 Churn Crusher 💰</h1>
  <h3 style="color: #E85D24;">Predicting Telecom Customer Exit with Machine Learning</h3>
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=120&section=header" width="100%" />
</div>
""", unsafe_allow_html=True)

# Navigation
st.sidebar.title("🕵️ Step-by-Step Flow")
step = st.sidebar.selectbox("Jump to Section:", [
    "1. Setup & Tools",
    "2. Data Loading & Inspection",
    "3. Visual Forensics (EDA)",
    "4. Data Prep & Encoding",
    "5. SMOTE Wizardry",
    "6. Model Showdown",
    "7. Verdict & Evaluations"
])

# --- Content ---

if step == "1. Setup & Tools":
    st.header("Step 1: Loading the Heavy Artillery")
    if st.button("🚀 Deploy Libraries"):
        warnings.filterwarnings('ignore')
        sns.set_theme(style="whitegrid", palette="viridis")
        st.success("✅ All systems go! AI libraries deployed.")
        st.code("""
import pandas as pd
import seaborn as sns
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
        """)

elif step == "2. Data Loading & Inspection":
    st.header("Step 2: Unboxing the Data")
    if st.button("📂 Load churn.csv"):
        try:
            df = pd.read_csv("churn.csv")
            st.session_state.df = df
            st.write(f"Dataset comprises **{df.shape[0]}** customers across **{df.shape[1]}** features.")
            st.dataframe(df.head())
            
            st.markdown("### ⚖️ Target Imbalance Check")
            churn_counts = df['Churn'].value_counts(normalize=True) * 100
            st.write(f"Churn Status: No ({churn_counts['No']:.1f}%) vs Yes ({churn_counts['Yes']:.1f}%)")
            st.warning("⚠️ Notice the imbalance! This is why we'll need SMOTE later!")
        except Exception as e:
            st.error(f"Error: {e}. Ensure churn.csv is in the project root.")

elif step == "3. Visual Forensics (EDA)":
    st.header("Step 3: Visual Forensics (EDA)")
    if 'df' not in st.session_state:
        st.warning("Please load data in Step 2 first.")
    else:
        df = st.session_state.df
        if st.button("📊 Show Distribution Trio"):
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            features = ['tenure', 'MonthlyCharges', 'TotalCharges']
            for i, col in enumerate(features):
                sns.kdeplot(data=df, x=col, hue='Churn', shade=True, ax=axes[i], palette=['#1D9E75','#E85D24'])
                axes[i].set_title(f'Distribution of {col} by Churn', fontweight='bold')
            st.pyplot(fig)
            st.info("💡 Insight: Churned customers (Orange) are concentrated at low tenure and high monthly charges!")

        if st.button("📜 The Contract Curse"):
            fig, ax = plt.subplots(figsize=(10,6))
            sns.histplot(data=df, x='Contract', hue='Churn', multiple='stack', shrink=.8, palette=['#1D9E75','#E85D24'], ax=ax)
            ax.set_title("Churn by Contract Type", fontweight='bold')
            st.pyplot(fig)

elif step == "4. Data Prep & Encoding":
    st.header("Step 4: Data Scrubbing & Prep")
    if 'df' not in st.session_state:
        st.warning("Please load data in Step 2.")
    else:
        if st.button("✨ Scrub Data"):
            df = st.session_state.df.copy()
            df.drop('customerID', axis=1, inplace=True, errors='ignore')
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
            
            mapping = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
            cols_to_map = ['Partner','Dependents','PhoneService','PaperlessBilling','Churn', 'gender']
            for col in cols_to_map:
                if col in df.columns: df[col] = df[col].map(mapping).fillna(df[col])
            
            cat_cols = ['MultipleLines','InternetService','OnlineSecurity', 'OnlineBackup','DeviceProtection','TechSupport', 'StreamingTV','StreamingMovies','Contract','PaymentMethod']
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
            
            st.session_state.df_prepared = df
            st.success(f"🚀 Dataset expanded to {df.shape[1]} columns. Ready for ML!")
            st.dataframe(df.head())

elif step == "5. SMOTE Wizardry":
    st.header("Step 5: The SMOTE Wizardry 🧙‍♂️")
    if 'df_prepared' not in st.session_state:
        st.warning("Please scrub data in Step 4.")
    else:
        if st.button("⚖️ Apply SMOTE"):
            df = st.session_state.df_prepared
            X = df.drop('Churn', axis=1)
            y = df['Churn'].astype(int)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            
            scaler = StandardScaler()
            st.session_state.X_train_res = X_train_res
            st.session_state.y_train_res = y_train_res
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.X_train_sc = scaler.fit_transform(X_train_res)
            st.session_state.X_test_sc = scaler.transform(X_test)
            st.session_state.feature_names = X.columns
            
            st.success("Training set balanced with SMOTE and scaled!")
            col1, col2 = st.columns(2)
            col1.metric("Before SMOTE (Churners)", int(y_train.sum()))
            col2.metric("After SMOTE (Churners)", int(y_train_res.sum()))

elif step == "6. Model Showdown":
    st.header("Step 6: The Model Showdown")
    if 'X_train_res' not in st.session_state:
        st.warning("Please apply SMOTE in Step 5.")
    else:
        if st.button("🏎️ Train Titans (LR, RF, XGB)"):
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            }
            results_data = []
            final_models = {}
            for name, model in models.items():
                train_x = st.session_state.X_train_sc if name == "Logistic Regression" else st.session_state.X_train_res
                test_x = st.session_state.X_test_sc if name == "Logistic Regression" else st.session_state.X_test
                model.fit(train_x, st.session_state.y_train_res)
                proba = model.predict_proba(test_x)[:,1]
                auc = roc_auc_score(st.session_state.y_test, proba)
                results_data.append({"Model": name, "AUC-ROC": round(auc, 4)})
                final_models[name] = {"model": model, "proba": proba, "preds": model.predict(test_x)}
            
            st.session_state.final_models = final_models
            st.table(pd.DataFrame(results_data))

elif step == "7. Verdict & Evaluations":
    st.header("Step 7: The Verdict (Evaluation)")
    if 'final_models' not in st.session_state:
        st.warning("Please train models in Step 6.")
    else:
        if st.button("🏆 Analyze Champion"):
            fig, axes = plt.subplots(1, 3, figsize=(20, 5))
            for i, (name, res) in enumerate(st.session_state.final_models.items()):
                cm = confusion_matrix(st.session_state.y_test, res['preds'])
                disp = ConfusionMatrixDisplay(cm, display_labels=['Stay', 'Churn'])
                disp.plot(ax=axes[i], cmap='YlGn', colorbar=False)
                axes[i].set_title(f"{name}\\nAcc: {(np.sum(np.diag(cm))/np.sum(cm)):.2%}")
            st.pyplot(fig)
            
            st.markdown("### 🏆 Feature Importance (Random Forest)")
            rf = st.session_state.final_models['Random Forest']['model']
            importances = pd.Series(rf.feature_importances_, index=st.session_state.feature_names).sort_values(ascending=False).head(10)
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.barplot(x=importances, y=importances.index, palette='magma', ax=ax2)
            st.pyplot(fig2)
            st.success("Business Recommendation: Focus on High Monthly Charges and Contract terms!")
