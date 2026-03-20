<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=250&section=header&text=Customer%20Churn%20Prediction&fontSize=60&animation=fadeIn&fontAlignY=38&desc=Predicting%20Telecom%20Customer%20Leaving%20Patterns&descAlignY=51&descAlign=62" width="100%" />

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Powered-green?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.ai/)

<br/>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=20&pause=1000&color=1D9E75&center=true&vCenter=true&width=600&lines=Analyze+7,043+customer+profiles;Handle+class+imbalance+with+SMOTE;Compare+LogReg,+RF,+and+XGBoost;Identify+churn+drivers+with+precision!" alt="Typing SVG" />
</p>

</div>

---

## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Smilies/Star-Struck.png" alt="Star-Struck" width="25" height="25" /> Project Highlights

*   <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Smilies/Robot.png" alt="Robot" width="20" height="20" /> **Predictive Modeling**: Deep dive into Logistic Regression, Random Forest, and XGBoost.
*   <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Scroll.png" alt="Scroll" width="20" height="20" /> **Advanced EDA**: Comprehensive visual analysis of tenure, charges, and contract types using Seaborn.
*   <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Symbols/Triangular%20Ruler.png" alt="Triangular Ruler" width="20" height="20" /> **Class Imbalance**: Implementing **SMOTE** to handle imbalanced datasets (26% churn rate).
*   <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/High%20Voltage.png" alt="High Voltage" width="20" height="20" /> **Business Insights**: Identifying key churn factors like monthly charges and contract types.

<br/>

## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/File%20Folder.png" alt="File Folder" width="25" height="25" /> Workflow Directory

| Step | Description |
| :--- | :--- |
| <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Magnifying%20Glass%20Tilted%20Right.png" alt="Search" width="30" height="30" /> **EDA** | Understanding data distributions, correlations, and finding churn patterns. |
| <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Soap.png" alt="Clean" width="30" height="30" /> **Data Cleaning** | Dropping IDs, fixing data types, and handling missing values in `TotalCharges`. |
| <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Hammer.png" alt="Train" width="30" height="30" /> **Modeling** | Training 3 distinct classifiers and handling class imbalance with SMOTE. |
| <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Activities/Trophy.png" alt="Result" width="30" height="30" /> **Evaluation** | Comparing AUC-ROC and Recall scores to pick the business-optimal model. |

<br/>

## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Rocket.png" alt="Rocket" width="30" height="30" /> Quick Start

Launch your churn prediction journey in minutes!

### <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Clipboard.png" alt="Clipboard" width="25" height="25" /> Prerequisites

*   [Python 3.8+](https://python.org/)
*   [Jupyter Notebook / VS Code](https://code.visualstudio.com/)
*   **Kaggle Dataset**: Download [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) CSV.

---

### <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Hammer%20and%20Wrench.png" alt="Setup" width="25" height="25" /> Local Installation

<img align="right" src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Laptop.png" width="150" height="150" alt="Laptop"/>

1. **Clone & Navigate:**
   ```bash
   git clone https://github.com/SubashSK777/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction
   ```

2. **Install Libraries:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
   ```

3. **Prepare Data:**
   Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the root directory and rename to `churn.csv`.

4. **Launch Notebook:**
   ```bash
   jupyter notebook churn.ipynb
   ```

<br/>

## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Handshakes/Handshake.png" alt="Results" width="30" height="30" /> Model Performance

| Model | AUC-ROC | Recall (Churn) |
| :--- | :--- | :--- |
| **Logistic Regression** | 0.808 | 63.6% |
| **Random Forest** | 0.817 | 58.3% |
| **XGBoost Classifier** | 0.810 | 57.5% |

*Note: Results obtained using SMOTE on training data and StandardScaler.*

---

## 📈 Key Findings
- **Contract Type**: Month-to-month contracts have the highest churn rate (~42%).
- **Tenure**: Customers with less than 12 months tenure are at high risk.
- **Charges**: Higher Monthly Charges correlate with higher churn probability.

<br/>

<div align="center">
  <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Rocket.png" alt="Rocket" width="50" height="50" />
  <h3>Built with ❤️ for Data Enthusiasts</h3>
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=footer" width="100%"/>
</div>
