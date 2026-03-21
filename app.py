import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import warnings

# Page Config
st.set_page_config(page_title="Churn Prediction App", layout="wide")

# Custom CSS (Matched with House Price UI)
st.markdown("""
    <style>
    .main { background-color: #050505; color: #ffffff; }
    .stButton>button {
        width: 100%; border-radius: 5px; height: 3em;
        background-color: #1D9E75; color: white;
        font-weight: bold; border: none; transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #167d5c; box-shadow: 0 4px 15px rgba(29, 158, 117, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# Header (Matched UI Style)
st.markdown("<h1 style='text-align: center; color: #1D9E75;'>💰 Churn Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predicting Telecom Customer Exit with Machine Learning Workflow.</p>", unsafe_allow_html=True)

# Navigation setup
steps = [
    "1. Setup & Tools",
    "2. Data Loading & Inspection",
    "3. Visual Forensics (EDA)",
    "4. Data Prep & Encoding",
    "5. SMOTE Wizardry",
    "6. Model Showdown",
    "7. Verdict & Evaluations"
]

if 'step_index' not in st.session_state:
    st.session_state.step_index = 0

def next_step():
    st.session_state.step_index = min(st.session_state.step_index + 1, len(steps) - 1)

step = st.sidebar.radio("Go to Step:", steps, index=st.session_state.step_index)
st.session_state.step_index = steps.index(step)

if 'df' not in st.session_state: st.session_state.df = None

# --- Content ---

if step == "1. Setup & Tools":
    st.header("Step 1: Loading the Heavy Artillery")
    if st.button("🚀 Deploy Libraries"):
        warnings.filterwarnings('ignore')
        sns.set_theme(style="whitegrid", palette="viridis")
        with st.expander("🔍 Deployment Details", expanded=False):
            st.success("✅ All systems go! AI libraries deployed.")
            st.code("import pandas as pd\nfrom xgboost import XGBClassifier\nfrom imblearn.over_sampling import SMOTE")
    
    if st.button("➡️ Next Part", key="next_1"):
        next_step(); st.rerun()

elif step == "2. Data Loading & Inspection":
    st.header("Step 2: Unboxing the Data")
    if st.button("📂 Load Dataset"):
        try:
            df = pd.read_csv("churn.csv")
            st.session_state.df = df
            with st.expander("📊 Data Overview", expanded=False):
                st.write(f"Dataset comprises **{df.shape[0]}** customers.")
                st.dataframe(df.head())
                churn_counts = df['Churn'].value_counts(normalize=True) * 100
                st.write(f"Churn Stat: No ({churn_counts['No']:.1f}%) vs Yes ({churn_counts['Yes']:.1f}%)")
        except Exception as e:
            st.error(f"Error: {e}. Ensure churn.csv is in the project root.")

    if st.session_state.df is not None:
        if st.button("➡️ Next Part", key="next_2"):
            next_step(); st.rerun()

elif step == "3. Visual Forensics (EDA)":
    st.header("Step 3: Visual Forensics (EDA)")
    if st.session_state.df is None:
        st.warning("Please load data in Step 2 first.")
    else:
        if st.button("📊 Show Distribution Trio"):
            with st.expander("📉 Feature Distributions", expanded=False):
                df = st.session_state.df.copy()
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                features = ['tenure', 'MonthlyCharges', 'TotalCharges']
                for i, col in enumerate(features):
                    sns.kdeplot(data=df, x=col, hue='Churn', shade=True, ax=axes[i], palette=['#1D9E75','#E85D24'])
                    axes[i].set_title(f'{col} Distribution')
                st.pyplot(fig)

        if st.button("📜 The Contract Curse"):
            with st.expander("📜 Contract Type Analysis", expanded=False):
                fig, ax = plt.subplots(figsize=(10,6))
                sns.histplot(data=st.session_state.df, x='Contract', hue='Churn', multiple='stack', shrink=.8, palette=['#1D9E75','#E85D24'], ax=ax)
                st.pyplot(fig)
        
        if st.button("➡️ Next Part", key="next_3"):
            next_step(); st.rerun()

elif step == "4. Data Prep & Encoding":
    st.header("Step 4: Data Scrubbing & Prep")
    if st.session_state.df is None:
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
            with st.expander("✨ Prep Result", expanded=False):
                st.success(f"🚀 Dataset expanded to {df.shape[1]} columns. Ready for ML!")
                st.dataframe(df.head())

        if 'df_prepared' in st.session_state:
            if st.button("➡️ Next Part", key="next_4"):
                next_step(); st.rerun()

elif step == "5. SMOTE Wizardry":
    st.header("Step 5: The SMOTE Wizardry 🧙‍♂️")
    if 'df_prepared' not in st.session_state:
        st.warning("Please scrub data in Step 4.")
    else:
        if st.button("⚖️ Apply SMOTE"):
            with st.expander("⚖️ Balancing Details", expanded=False):
                df = st.session_state.df_prepared
                X = df.drop('Churn', axis=1)
                y = df['Churn'].astype(int)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                smote = SMOTE(random_state=42)
                X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
                scaler = StandardScaler()
                st.session_state.X_train_res, st.session_state.y_train_res = X_train_res, y_train_res
                st.session_state.X_test, st.session_state.y_test = X_test, y_test
                st.session_state.X_train_sc = scaler.fit_transform(X_train_res)
                st.session_state.X_test_sc = scaler.transform(X_test)
                st.session_state.feature_names = X.columns
                st.success("Training set balanced with SMOTE and scaled!")
                st.write(f"Samples after SMOTE: {len(y_train_res)}")

        if 'X_train_res' in st.session_state:
            if st.button("➡️ Next Part", key="next_5"):
                next_step(); st.rerun()

elif step == "6. Model Showdown":
    st.header("Step 6: The Model Showdown")
    if 'X_train_res' not in st.session_state:
        st.warning("Please apply SMOTE in Step 5.")
    else:
        if st.button("🏎️ Train Models"):
            with st.expander("🏎️ Training Performance", expanded=False):
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                }
                results_data = []; final_models = {}
                for name, model in models.items():
                    tx = st.session_state.X_train_sc if name == "Logistic Regression" else st.session_state.X_train_res
                    tsx = st.session_state.X_test_sc if name == "Logistic Regression" else st.session_state.X_test
                    model.fit(tx, st.session_state.y_train_res)
                    proba = model.predict_proba(tsx)[:,1]
                    auc = roc_auc_score(st.session_state.y_test, proba)
                    results_data.append({"Model": name, "AUC-ROC": round(auc, 4)})
                    final_models[name] = {"model": model, "proba": proba, "preds": model.predict(tsx)}
                st.session_state.final_models = final_models
                st.table(pd.DataFrame(results_data))

        if 'final_models' in st.session_state:
            if st.button("➡️ Next Part", key="next_6"):
                next_step(); st.rerun()

elif step == "7. Verdict & Evaluations":
    st.header("Step 7: The Verdict (Evaluation)")
    if 'final_models' not in st.session_state:
        st.warning("Please train models in Step 6.")
    else:
        if st.button("🏆 Analyze Champion"):
            with st.expander("🥇 Evaluation Results", expanded=False):
                fig, axes = plt.subplots(1, 3, figsize=(20, 5))
                for i, (name, res) in enumerate(st.session_state.final_models.items()):
                    cm = confusion_matrix(st.session_state.y_test, res['preds'])
                    disp = ConfusionMatrixDisplay(cm, display_labels=['Stay', 'Churn'])
                    disp.plot(ax=axes[i], cmap='YlGn', colorbar=False)
                    axes[i].set_title(f"{name}")
                st.pyplot(fig)
        
        st.success("🎉 Exploration complete!")
        if st.button("🚀 Restart Process"):
            st.session_state.step_index = 0
            st.rerun()
